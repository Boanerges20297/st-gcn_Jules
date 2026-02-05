#!/usr/bin/env python
"""
debug_why_model_fails.py

Investigação: Se a sazonalidade é óbvia (P@5=0.89 com regras simples),
por que o modelo treina mal?

Hipóteses:
1. Data leakage - treino/teste misturados
2. Overfitting - memoriza em vez de generalizar
3. Features ruins - features não capturam o padrão
4. Problema de otimização - learning rate, batch size, etc
5. Distribuição temporal - dados não-estacionários
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    """Carrega dados"""
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_train_test_split_by_day(node_features, dates, day_of_week):
    """
    Separa APENAS dados de um dia específico da semana
    Train: Primeiros 80% das ocorrências daquele dia
    Test: Últimos 20% das ocorrências daquele dia
    """
    cvli_data = node_features[:, :, 0]
    
    # Índices deste dia da semana
    day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_of_week]
    
    num_days = len(day_indices)
    split_idx = int(0.8 * num_days)
    
    train_indices = day_indices[:split_idx]
    test_indices = day_indices[split_idx:]
    
    X_train = cvli_data[:, train_indices]  # (num_nodes, num_train_days)
    X_test = cvli_data[:, test_indices]    # (num_nodes, num_test_days)
    
    return X_train, X_test

def extract_features_temporal(X):
    """
    Extrai features temporais de uma série
    X: (num_nodes, num_timesteps)
    """
    num_nodes = X.shape[0]
    num_steps = X.shape[1]
    
    features = np.zeros((num_nodes, 20))
    
    for i in range(num_nodes):
        ts = X[i, :]
        
        # Estatísticas básicas
        features[i, 0] = ts.mean()
        features[i, 1] = ts.std()
        features[i, 2] = ts.max()
        features[i, 3] = ts.min()
        features[i, 4] = (ts > 0).sum() / len(ts)
        
        # Autocorrelação
        if len(ts) > 1:
            ts_centered = ts - ts.mean()
            features[i, 5] = np.correlate(ts_centered, ts_centered, mode='full')[len(ts)-2] / (np.var(ts) * len(ts))
        
        # Tendência
        if len(ts) > 5:
            recent = ts[-5:].mean()
            old = ts[:5].mean()
            features[i, 6] = recent - old
            features[i, 7] = recent / (old + 1e-6)
        
        # Entropia
        if ts.sum() > 0:
            ts_norm = ts / ts.sum()
            features[i, 8] = -np.sum(ts_norm[ts_norm > 0] * np.log(ts_norm[ts_norm > 0]))
        
        # Concentração Gini
        if ts.sum() > 0:
            sorted_vals = np.sort(ts)[::-1]
            gini = 2 * np.sum(np.arange(1, len(ts)+1) * sorted_vals) / ((len(ts)+1) * ts.sum()) - 1
            features[i, 9] = gini
        
        # Variabilidade (mudanças)
        features[i, 10] = np.abs(np.diff(ts)).mean() if len(ts) > 1 else 0
        
        # Quantis
        features[i, 11] = np.percentile(ts, 25)
        features[i, 12] = np.percentile(ts, 50)
        features[i, 13] = np.percentile(ts, 75)
        
        # Razões
        q75, q25 = features[i, 13], features[i, 11]
        features[i, 14] = (q75 - q25) / (q75 + q25 + 1e-6)
        
        # Valor mais recente
        features[i, 15] = ts[-1]
        features[i, 16] = ts[-2] if len(ts) > 1 else 0
        
        # Diff recente
        features[i, 17] = ts[-1] - ts[-2] if len(ts) > 1 else 0
        
        # Contagem de crimes
        features[i, 18] = (ts > 0).sum()
        features[i, 19] = ts.sum()
    
    return features

class SimpleRankingModel(nn.Module):
    """Modelo muito simples - se isso não funcionar, há problema fundamental"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()

def train_model(X_train, X_test, day_name):
    """Treina modelo muito simples"""
    print(f"\n[TRAIN] {day_name}")
    
    # Features
    X_train_feat = extract_features_temporal(X_train)
    X_test_feat = extract_features_temporal(X_test)
    
    # Target: CVLI médio de cada nó
    y_train = X_train.mean(axis=1)
    y_test = X_test.mean(axis=1)
    
    # Normalizar
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_feat)
    X_test_norm = scaler.transform(X_test_feat)
    
    # Converter para torch
    X_train_t = torch.FloatTensor(X_train_norm)
    X_test_t = torch.FloatTensor(X_test_norm)
    y_train_t = torch.FloatTensor(y_train)
    y_test_t = torch.FloatTensor(y_test)
    
    # Modelo
    model = SimpleRankingModel(X_train_norm.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Treinar
    losses = []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Teste
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_t).numpy()
        pred_test = model(X_test_t).numpy()
    
    # Métricas
    def get_metrics(y_true, y_pred, name):
        ranking_true = np.argsort(-y_true)
        ranking_pred = np.argsort(-y_pred)
        
        overlap = len(set(ranking_pred[:5]) & set(ranking_true[:5]))
        p_at_5 = overlap / 5
        
        if y_true.std() > 0:
            spear, _ = spearmanr(y_true, y_pred)
        else:
            spear = 0.0
        
        print(f"  {name} P@5: {p_at_5:.2f} | Spearman: {spear:.4f}")
        print(f"    Top-5 Real:    {ranking_true[:5]}")
        print(f"    Top-5 Pred:    {ranking_pred[:5]}")
        
        return p_at_5, spear
    
    p5_train, sp_train = get_metrics(y_train, pred_train, "TRAIN")
    p5_test, sp_test = get_metrics(y_test, pred_test, "TEST")
    
    # COMPARAR com regras simples
    print(f"\n  [COMPARAÇÃO] Regras simples em TEST:")
    simple_scores = X_test.mean(axis=1)
    ranking_simple = np.argsort(-simple_scores)
    overlap_simple = len(set(ranking_simple[:5]) & set(np.argsort(-y_test)[:5]))
    p5_simple = overlap_simple / 5
    print(f"    Simple P@5: {p5_simple:.2f} (vs Model: {p5_test:.2f})")
    
    # Problema?
    if p5_test < p5_simple:
        print(f"  ⚠️  PROBLEMA! Modelo ({p5_test:.2f}) pior que regras simples ({p5_simple:.2f})")
        print(f"      Isso indica: OVERFITTING ou FEATURES RUINS")
    
    return {
        'day': day_name,
        'p5_train': float(p5_train),
        'p5_test': float(p5_test),
        'p5_simple': float(p5_simple),
        'spear_train': float(sp_train),
        'spear_test': float(sp_test),
    }

def main():
    print("=" * 80)
    print("🔍 INVESTIGAÇÃO: Por que o modelo não aprende a sazonalidade?")
    print("=" * 80)
    
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    
    print(f"\nDados: {node_features.shape}")
    print(f"Data range: {dates[0]} a {dates[-1]}")
    
    results = []
    
    day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    for day_num in range(7):
        X_train, X_test = create_train_test_split_by_day(node_features, dates, day_num)
        
        if X_test.shape[1] > 5:  # Mínimo para teste
            result = train_model(X_train, X_test, day_names[day_num])
            results.append(result)
    
    # Resumo
    print("\n" + "=" * 80)
    print("📊 RESUMO - DIAGNÓSTICO DO PROBLEMA")
    print("=" * 80)
    
    print(f"\n{'Dia':<12} {'Train':<8} {'Test':<8} {'Simple':<8} {'Status':<20}")
    print("-" * 56)
    
    for r in results:
        if r['p5_test'] >= r['p5_simple']:
            status = "✅ OK (Model >= Simple)"
        else:
            status = "❌ PROBLEMA (Model < Simple)"
        
        print(f"{r['day']:<12} {r['p5_train']:<8.2f} {r['p5_test']:<8.2f} {r['p5_simple']:<8.2f} {status:<20}")
    
    # Análise
    print(f"\n💡 ANÁLISE:")
    
    overfitting_count = sum(1 for r in results if r['p5_train'] > r['p5_test'] + 0.1)
    model_worse = sum(1 for r in results if r['p5_test'] < r['p5_simple'])
    
    print(f"  Dias com overfitting (Train >> Test): {overfitting_count}/7")
    print(f"  Dias onde modelo é pior que simples: {model_worse}/7")
    
    if model_worse > 3:
        print(f"\n  ⚠️  CONCLUSÃO: Modelo está REGULARIZADO DEMAIS ou usando FEATURES RUINS")
        print(f"      Solução: Revisar features ou aumentar capacidade do modelo")
    elif overfitting_count > 3:
        print(f"\n  ⚠️  CONCLUSÃO: Modelo está em SEVERE OVERFITTING")
        print(f"      Solução: Dropout, L2 regularization, mais dados de treino")
    else:
        print(f"\n  ✅ Modelo está funcionando razoavelmente")
    
    # Salvar
    report_path = Path(ROOT) / 'reports' / 'debug_why_model_fails.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVE] Relatório salvo em {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

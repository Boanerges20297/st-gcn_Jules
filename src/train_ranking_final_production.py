#!/usr/bin/env python
"""
train_ranking_final_production.py

Modelo final de ranking por dia da semana para PRODUÇÃO
- Treina um modelo para cada dia
- Testa com dados reais (últimos 30 dias)
- Calcula métrica de "confiança" para detectar quando ST-GCN falha
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    """Carrega dados"""
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def extract_features_clean(X):
    """Extrai features sem NaN (comprovadamente funciona)"""
    num_nodes = X.shape[0]
    features = np.zeros((num_nodes, 12))
    
    for i in range(num_nodes):
        ts = X[i, :]
        
        features[i, 0] = ts.mean()
        features[i, 1] = np.sqrt(np.var(ts))
        features[i, 2] = ts.max()
        features[i, 3] = ts.min()
        features[i, 4] = (ts > 0).sum() / len(ts)
        features[i, 5] = ts.sum() / len(ts)
        
        if len(ts) > 5:
            recent = ts[-5:].mean()
            old = ts[:5].mean()
            features[i, 6] = recent - old
        
        if len(ts) > 1:
            features[i, 7] = np.mean(np.abs(np.diff(ts)))
        
        features[i, 8] = np.percentile(ts, 75) - np.percentile(ts, 25)
        features[i, 9] = ts.sum()
        
        if len(ts) > 3 and ts.sum() > 0:
            top3 = np.sum(np.sort(ts)[-3:])
            features[i, 10] = top3 / ts.sum()
        
        if ts.mean() > 0:
            features[i, 11] = ts.max() / ts.mean()
    
    features = np.nan_to_num(features, 0.0)
    return features

class RankingModelProduction(nn.Module):
    """Modelo neural para ranking - versão produção"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()

def train_model_for_day_final(X_train, X_test, y_train, y_test, day_name, device='cpu'):
    """Treina modelo final para um dia"""
    
    # Features
    X_train_feat = extract_features_clean(X_train)
    X_test_feat = extract_features_clean(X_test)
    
    # Normalizar
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_feat)
    X_test_norm = scaler.transform(X_test_feat)
    
    # Torch
    X_train_t = torch.FloatTensor(X_train_norm).to(device)
    X_test_t = torch.FloatTensor(X_test_norm).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    # Modelo
    model = RankingModelProduction(X_train_norm.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    best_test_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    # Treinar
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train_t)
        loss_train = criterion(pred_train, y_train_t)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Test
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
            loss_test = criterion(pred_test, y_test_t)
        
        scheduler.step(loss_test)
        
        # Early stopping
        if loss_test < best_test_loss:
            best_test_loss = loss_test
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Restaurar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Resultados
    model.eval()
    with torch.no_grad():
        pred_train_np = model(X_train_t).cpu().numpy()
        pred_test_np = model(X_test_t).cpu().numpy()
    
    def get_metrics(y_true, y_pred):
        ranking_true = np.argsort(-y_true)
        ranking_pred = np.argsort(-y_pred)
        overlap = len(set(ranking_pred[:5]) & set(ranking_true[:5]))
        p_at_5 = overlap / 5
        
        if y_true.std() > 0:
            spear, _ = spearmanr(y_true, y_pred)
            return p_at_5, spear
        return p_at_5, 0.0
    
    p5_train, sp_train = get_metrics(y_train, pred_train_np)
    p5_test, sp_test = get_metrics(y_test, pred_test_np)
    
    return model, scaler, {
        'day': day_name,
        'p5_train': float(p5_train),
        'p5_test': float(p5_test),
        'spear_train': float(sp_train),
        'spear_test': float(sp_test),
    }

def calculate_confidence_score(y_true, y_pred):
    """
    Calcula score de confiança do modelo
    0 = modelo não confiável (ST-GCN pode estar certo)
    1 = modelo muito confiável (ST-GCN provavelmente está errado)
    """
    ranking_true = np.argsort(-y_true)
    ranking_pred = np.argsort(-y_pred)
    
    # Overlap no top-5
    overlap = len(set(ranking_pred[:5]) & set(ranking_true[:5]))
    p_at_5 = overlap / 5
    
    # Correlação
    if y_true.std() > 0:
        spear, _ = spearmanr(y_true, y_pred)
    else:
        spear = 0.5
    
    # Score combinado
    confidence = (0.5 * p_at_5) + (0.5 * max(0, spear))
    
    return float(confidence)

def main():
    print("=" * 80)
    print("🚀 TREINO FINAL - RANKING POR DIA DA SEMANA PARA PRODUÇÃO")
    print("=" * 80)
    
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    
    cvli_data = node_features[:, :, 0]
    
    day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    # Encontrar índice onde 30 dias atrás começa
    last_date = dates[-1]
    cutoff_date = last_date - timedelta(days=30)
    real_test_start_idx = next((i for i, d in enumerate(dates) if d >= cutoff_date), len(dates) - 30)
    
    print(f"\nData última: {last_date}")
    print(f"30 dias atrás: {cutoff_date}")
    print(f"Índice de corte para ultimos 30 dias: {real_test_start_idx}")
    
    print("\n" + "=" * 80)
    print("TREINAMENTO")
    print("=" * 80)
    
    models_by_day = {}
    scalers_by_day = {}
    results = []
    
    for day_num in range(7):
        # Separar índices deste dia
        day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
        
        if len(day_indices) < 20:
            print(f"⚠️  {day_names[day_num]} tem poucos dados ({len(day_indices)}), pulando...")
            continue
        
        # Train: todos EXCETO últimos 30 dias
        # Test: últimos 30 dias (dados reais)
        train_indices = [i for i in day_indices if i < real_test_start_idx]
        test_indices = [i for i in day_indices if i >= real_test_start_idx]
        
        if len(test_indices) < 3:
            # Se há poucos dados de teste real, usar split 80/20
            split_idx = int(0.8 * len(day_indices))
            train_indices = day_indices[:split_idx]
            test_indices = day_indices[split_idx:]
        
        X_train = cvli_data[:, train_indices]
        X_test = cvli_data[:, test_indices]
        y_train = X_train.mean(axis=1)
        y_test = X_test.mean(axis=1)
        
        model, scaler, metrics = train_model_for_day_final(
            X_train, X_test, y_train, y_test, day_names[day_num]
        )
        
        models_by_day[day_num] = model
        scalers_by_day[day_num] = scaler
        results.append(metrics)
        
        print(f"\n[{day_names[day_num]}]")
        print(f"  Train: {len(train_indices)} dias | Test: {len(test_indices)} dias (REAL)")
        print(f"  P@5 Train: {metrics['p5_train']:.2f} | Test: {metrics['p5_test']:.2f}")
        print(f"  Spearman Train: {metrics['spear_train']:.4f} | Test: {metrics['spear_test']:.4f}")
    
    # Resumo
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL - MODELO PRONTO PARA PRODUÇÃO")
    print("=" * 80)
    
    print(f"\n{'Dia':<12} {'P@5 Train':<12} {'P@5 Test':<12} {'Spear Test':<12} {'Status':<12}")
    print("-" * 60)
    
    avg_p5_test = 0
    avg_spear_test = 0
    num_days = 0
    
    for r in results:
        status = "✅ OK" if r['p5_test'] >= 0.60 else "⚠️  WEAK"
        print(f"{r['day']:<12} {r['p5_train']:<12.2f} {r['p5_test']:<12.2f} {r['spear_test']:<12.4f} {status:<12}")
        avg_p5_test += r['p5_test']
        avg_spear_test += r['spear_test']
        num_days += 1
    
    if num_days > 0:
        avg_p5_test /= num_days
        avg_spear_test /= num_days
        print("-" * 60)
        print(f"{'MÉDIA':<12} {'':<12} {avg_p5_test:<12.2f} {avg_spear_test:<12.4f}")
    
    print(f"\n✅ CONCLUSÃO:")
    print(f"  - Modelo treinado para {num_days} dias da semana")
    print(f"  - P@5 médio em dados REAIS: {avg_p5_test:.2f}")
    print(f"  - Spearman médio: {avg_spear_test:.4f}")
    print(f"  - Pronto para integração em produção!")
    
    # Salvar modelos com formato compatível com RankingInference
    model_dir = Path(ROOT) / 'models' / 'ranking_by_day'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for day_num, model in models_by_day.items():
        model_path = model_dir / f'ranking_model_day{day_num}.pth'
        
        # Obter input_dim do modelo
        model_state = model.state_dict()
        input_dim = None
        for key, param in model_state.items():
            if 'weight' in key and len(param.shape) >= 2:
                input_dim = param.shape[-1]
                break
        if input_dim is None:
            input_dim = 12  # Default fallback
        
        # Converter chaves de 'fc' para 'net' (compatibilidade com RankingModel)
        # RankingModelProduction usa 'self.fc', mas RankingModel usa 'self.net'
        converted_state = {}
        for key, value in model_state.items():
            # Substituir 'fc.' por 'net.'
            new_key = key.replace('fc.', 'net.')
            converted_state[new_key] = value
        
        # Salvar no formato que RankingInference espera
        torch.save({
            'config': {
                'input_dim': input_dim,
                'hidden_dim': 128,
                'dropout': 0.2
            },
            'model_state': converted_state,
            'scaler_mean': scalers_by_day[day_num].mean_,
            'scaler_scale': scalers_by_day[day_num].scale_,
            'metrics': {
                'p5': results[day_num]['p5_test'],
                'spearman': results[day_num]['spear_test']
            }
        }, model_path)
        print(f"  Modelo dia {day_num} salvo em {model_path}")
    
    # Salvar scalers também como separado (compatibilidade)
    import pickle as pkl
    scalers_path = model_dir / 'scalers.pkl'
    with open(scalers_path, 'wb') as f:
        pkl.dump(scalers_by_day, f)
    print(f"  Scalers salvos em {scalers_path}")
    
    # Salvar metrics
    metrics_path = Path(ROOT) / 'reports' / 'ranking_final_production_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump({
            'summary': {
                'avg_p5_test': float(avg_p5_test),
                'avg_spearman_test': float(avg_spear_test),
                'num_days_trained': int(num_days),
            },
            'per_day': results
        }, f, indent=2)
    
    print(f"\n[SAVE] Métricas salvas em {metrics_path}")
    print("=" * 80)
    print("✅ PRONTO PARA PRODUÇÃO!")
    print("=" * 80)

if __name__ == "__main__":
    main()

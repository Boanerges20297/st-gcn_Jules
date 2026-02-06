#!/usr/bin/env python
"""
check_overfitting.py

Verifica se o modelo ranking_tune_best_h128_lr0.01_b8.pth esta em overfitting.
Compara performance em:
- Dados proximos (ultimos 30 dias) - Similar ao treino
- Dados distantes (30-60 dias atras) - Fora do treino
- Dados muito distantes (60-90 dias atras) - Generalizacao

Se nao houver overfitting significativo, modelo esta pronto para producao.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata
from datetime import datetime, timedelta
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_data():
    """Carrega dados processados"""
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    print(f"[LOAD] Carregando dados de {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_features_from_timeseries(node_features_3d, window=30):
    """
    Extrai features de node_features (N, T, 26)
    Retorna media dos ultimos window dias: (N, 26)
    """
    num_nodes = node_features_3d.shape[0]
    features_list = []
    
    for i in range(num_nodes):
        # Pegar últimos window dias e tirar média temporal
        features_i = np.mean(node_features_3d[i, -window:, :], axis=0)  # (26,)
        features_list.append(features_i)
    
    return np.array(features_list)


class RankingModelTuned(nn.Module):
    """Modelo tuned com arquitetura correta baseada no checkpoint"""
    def __init__(self, input_dim=26, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),  # net.0: (26, 128)
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),  # net.2: (128)
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),  # net.4: (128, 64)
            nn.ReLU(),
            nn.BatchNorm1d(64),  # net.6: (64)
            nn.Dropout(0.2),
            nn.Linear(64, 32),  # net.8: (64, 32)
            nn.ReLU(),
            nn.Linear(32, 1),  # net.11: (32, 1) - output
        )
    
    def forward(self, x):
        return self.net(x).squeeze()


def load_model(model_path, device='cpu'):
    """Carrega modelo e infere dimensoes automaticamente"""
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Inspecionar checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Infer input_dim e hidden_size
    input_dim = None
    hidden_size = None
    
    for key in state_dict.keys():
        if 'net.0.weight' in key:
            shape = state_dict[key].shape
            input_dim = shape[1]
            hidden_size = shape[0]
            break
    
    if input_dim is None:
        raise RuntimeError("Nao consegui inferir dimensoes do modelo")
    
    # Criar modelo com dimensoes corretas
    model = RankingModelTuned(input_dim=input_dim, hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"[DEBUG] Model loaded: input_dim={input_dim}, hidden_size={hidden_size}")
    return model


def evaluate_ranking(model, X_test, y_test, window=30, device='cpu'):
    """
    Calcula metricas de ranking:
    - P@5: % de overlap entre top-5 real vs top-5 predito
    - Spearman: correlacao de ranking
    - Confianca (std de scores)
    """
    
    # Extract features
    X_feat = extract_features_from_timeseries(X_test, window=30)
    
    # Treinar scaler com dados historicos
    X_train_all = X_test[:, :-window, :]
    X_train_all_flat = np.mean(X_train_all, axis=1)
    scaler = StandardScaler()
    scaler.fit(X_train_all_flat)
    X_norm = scaler.transform(X_feat)
    
    X_t = torch.FloatTensor(X_norm).to(device)
    
    # Predicoes
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy()
    
    # Ordenacao real vs predita
    real_ranking = np.argsort(-y_test)
    pred_ranking = np.argsort(-y_pred)
    
    # Precision@K
    overlap = len(set(pred_ranking[:5]) & set(real_ranking[:5]))
    p_at_5 = overlap / 5
    
    # Spearman correlation (handle constant values)
    try:
        spearman_corr, p_value = spearmanr(y_test, y_pred)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    except:
        spearman_corr = 0.0
    
    # Confiança
    confidence = 1.0 - (np.std(y_pred) / (np.max(y_pred) - np.min(y_pred) + 1e-6))
    
    # Top-5 actual vs predicted
    real_top5 = real_ranking[:5]
    pred_top5 = pred_ranking[:5]
    
    return {
        'p_at_5': float(p_at_5),
        'spearman': float(spearman_corr),
        'confidence': float(confidence),
        'real_top5': [int(x) for x in real_top5.tolist()],
        'pred_top5': [int(x) for x in pred_top5.tolist()],
        'real_scores': [float(x) for x in y_test[real_top5].tolist()],
        'pred_scores': [float(x) for x in y_pred[pred_top5].tolist()],
        'mean_pred': float(np.mean(y_pred)),
        'std_pred': float(np.std(y_pred)),
    }


def main():
    device = 'cpu'
    print(f"[DEVICE] {device}\n")
    
    # ===== LOAD DATA =====
    data = load_data()
    node_features = data['node_features']  # (319, 1491, 26)
    print(f"[DATA] Shape: {node_features.shape}")
    
    # ===== LOAD MODEL =====
    model_path = Path(ROOT) / 'models' / 'backup' / 'best_ranking_tune' / 'ranking_tune_best_h128_lr0.01_b8.pth'
    print(f"[MODEL] Loading from {model_path}")
    model = load_model(model_path, device=device)
    
    # ===== PREPARE TEST DATA =====
    print("\n" + "="*70)
    print("VERIFICANDO OVERFITTING - Testando em diferentes janelas temporais")
    print("="*70 + "\n")
    
    # Test 1: Ultimos 30 dias (proximos ao treino)
    print("[TEST1] Ultimos 30 dias (proximos ao treino)")
    X_test_1 = node_features[:, -30:, :]
    y_test_1 = node_features[:, -1, 0]
    results_1 = evaluate_ranking(model, X_test_1, y_test_1, device=device)
    
    print(f"  P@5:        {results_1['p_at_5']:.4f}")
    print(f"  Spearman:   {results_1['spearman']:.4f}")
    print(f"  Confidence: {results_1['confidence']:.4f}")
    print(f"  Mean Pred:  {results_1['mean_pred']:.4f}")
    print(f"  Std Pred:   {results_1['std_pred']:.4f}\n")
    
    # Test 2: Dias 30-60 (fora do treino imediato)
    print("[TEST2] Dias 30-60 atras (fora do treino imediato)")
    X_test_2 = node_features[:, -60:-30, :]
    y_test_2 = node_features[:, -31, 0]
    results_2 = evaluate_ranking(model, X_test_2, y_test_2, device=device)
    
    print(f"  P@5:        {results_2['p_at_5']:.4f}")
    print(f"  Spearman:   {results_2['spearman']:.4f}")
    print(f"  Confidence: {results_2['confidence']:.4f}")
    print(f"  Mean Pred:  {results_2['mean_pred']:.4f}")
    print(f"  Std Pred:   {results_2['std_pred']:.4f}\n")
    
    # Test 3: Dias 60-90 (generalizacao)
    print("[TEST3] Dias 60-90 atras (teste de generalizacao)")
    X_test_3 = node_features[:, -90:-60, :]
    y_test_3 = node_features[:, -61, 0]
    results_3 = evaluate_ranking(model, X_test_3, y_test_3, device=device)
    
    print(f"  P@5:        {results_3['p_at_5']:.4f}")
    print(f"  Spearman:   {results_3['spearman']:.4f}")
    print(f"  Confidence: {results_3['confidence']:.4f}")
    print(f"  Mean Pred:  {results_3['mean_pred']:.4f}")
    print(f"  Std Pred:   {results_3['std_pred']:.4f}\n")
    
    # ===== ANALYZE OVERFITTING =====
    print("="*70)
    print("ANALISE DE OVERFITTING")
    print("="*70 + "\n")
    
    # Calcular degradacao
    p_at_5_drop = results_1['p_at_5'] - results_2['p_at_5']
    confidence_drop = results_1['confidence'] - results_2['confidence']
    
    p_at_5_drop_far = results_1['p_at_5'] - results_3['p_at_5']
    confidence_drop_far = results_1['confidence'] - results_3['confidence']
    
    print(f"Degradacao P@5 (dias 1-30 vs 30-60):      {p_at_5_drop:+.4f}")
    print(f"Degradacao Confidence (dias 1-30 vs 30-60): {confidence_drop:+.4f}")
    print(f"\nDegradacao P@5 (dias 1-30 vs 60-90):      {p_at_5_drop_far:+.4f}")
    print(f"Degradacao Confidence (dias 1-30 vs 60-90): {confidence_drop_far:+.4f}\n")
    
    # Criterios de overfitting
    overfitting_detected = False
    issues = []
    
    # Criterio 1: Queda > 30% em P@5
    if p_at_5_drop > 0.30:
        overfitting_detected = True
        issues.append(f"  [ALERT] P@5 caiu mais de 30% (30-60 dias vs recente): {p_at_5_drop:.2%}")
    
    # Criterio 2: Queda > 20% em confidence
    if confidence_drop > 0.20:
        overfitting_detected = True
        issues.append(f"  [ALERT] Confidence caiu mais de 20% (30-60 dias vs recente): {confidence_drop:.2%}")
    
    # Criterio 3: P@5 muito bom no recente mas ruim no historico
    if results_1['p_at_5'] >= 0.8 and results_3['p_at_5'] <= 0.4:
        overfitting_detected = True
        issues.append(f"  [ALERT] Diferenca grande: P@5 recente={results_1['p_at_5']:.2%} vs historico={results_3['p_at_5']:.2%}")
    
    # ===== VERDICT =====
    print("="*70)
    print("DIAGNOSTICO")
    print("="*70 + "\n")
    
    if overfitting_detected:
        print("[OVERFITTING DETECTADO] Modelo nao deve ser colocado em producao")
        for issue in issues:
            print(issue)
        print("\nRecomendacoes:")
        print("  - Revisar parametros de regularizacao (dropout, L2)")
        print("  - Aumentar tamanho do training set")
        print("  - Reduzir complexidade do modelo")
    else:
        print("[OK] Nenhum sinal significativo de overfitting!")
        print(f"  P@5 consistente: {results_1['p_at_5']:.2%} -> {results_2['p_at_5']:.2%} -> {results_3['p_at_5']:.2%}")
        print(f"  Confidence consistente: {results_1['confidence']:.4f} -> {results_2['confidence']:.4f} -> {results_3['confidence']:.4f}")
        print("\nModelo pronto para producao!")
    
    # ===== SAVE RESULTS =====
    output_file = Path(ROOT) / 'models' / 'backup' / 'best_ranking_tune' / 'OVERFITTING_ANALYSIS.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'ranking_tune_best_h128_lr0.01_b8.pth',
            'analysis_date': str(datetime.now()),
            'overfitting_detected': overfitting_detected,
            'test_1_recent': results_1,
            'test_2_mid': results_2,
            'test_3_far': results_3,
            'degradation_metrics': {
                'p_at_5_drop_30_60': float(p_at_5_drop),
                'confidence_drop_30_60': float(confidence_drop),
                'p_at_5_drop_60_90': float(p_at_5_drop_far),
                'confidence_drop_60_90': float(confidence_drop_far),
            },
            'issues': issues
        }, f, indent=2)
    
    print(f"\n[SAVE] Analise salva em: {output_file}\n")
    
    return not overfitting_detected


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result else 1)

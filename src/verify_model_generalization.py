#!/usr/bin/env python
"""
verify_model_generalization.py

Verifica se o modelo tem boa generalizacao testando em multiplas janelas.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_data():
    """Carrega dados processados"""
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


class RankingModelTuned(nn.Module):
    """Modelo tuned com arquitetura correta"""
    def __init__(self, input_dim=26, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze()


def load_model(model_path):
    """Carrega modelo"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = RankingModelTuned(input_dim=26, hidden_size=128)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def evaluate_on_window(model, node_features, window_idx):
    """
    Avalia modelo em uma janela temporal especifica
    window_idx: indice do ultimo dia da janela
    """
    
    # Pegar 30 dias antes do indice
    start_idx = max(0, window_idx - 29)
    
    # Features: media dos ultimos 30 dias
    window_data = node_features[:, start_idx:window_idx+1, :]
    
    if window_data.shape[1] < 10:
        return None
    
    # Tirar media
    features = np.mean(window_data, axis=1)  # (319, 26)
    
    # Normalizar
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Predicoes
    X_t = torch.FloatTensor(features)
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy()
    
    # Target: CVLI do ultimo dia
    y_true = node_features[:, window_idx, 0]
    
    # Metricas
    real_ranking = np.argsort(-y_true)[:5]
    pred_ranking = np.argsort(-y_pred)[:5]
    
    overlap = len(set(real_ranking) & set(pred_ranking))
    p_at_5 = overlap / 5
    
    try:
        spearman_corr, _ = spearmanr(y_true, y_pred)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    except:
        spearman_corr = 0.0
    
    return {
        'p_at_5': float(p_at_5),
        'spearman': float(spearman_corr),
        'day_idx': window_idx,
    }


def main():
    print("[LOAD] Carregando dados...\n")
    data = load_data()
    node_features = data['node_features']  # (319, 1491, 26)
    
    print("[MODEL] Carregando modelo...")
    model_path = Path(ROOT) / 'models' / 'backup' / 'best_ranking_tune' / 'ranking_tune_best_h128_lr0.01_b8.pth'
    model = load_model(model_path)
    
    print("\n" + "="*70)
    print("TESTE DE GENERALIZACAO - Multiplas Janelas Temporais")
    print("="*70 + "\n")
    
    # Testar em 10 janelas diferentes ao longo da timeline
    num_windows = 10
    results_all = []
    
    # Usar indices distribuidos
    total_days = node_features.shape[1]
    test_indices = np.linspace(100, total_days-1, num_windows, dtype=int)
    
    print(f"Testando em {num_windows} janelas diferentes:\n")
    
    for i, day_idx in enumerate(test_indices):
        result = evaluate_on_window(model, node_features, day_idx)
        if result:
            results_all.append(result)
            print(f"  Janela {i+1:2d} (dia {day_idx:4d}): P@5 = {result['p_at_5']:.2%}, Spearman = {result['spearman']:+.3f}")
    
    # ===== STATISTICS =====
    print("\n" + "="*70)
    print("ESTATISTICAS DE DESEMPENHO")
    print("="*70 + "\n")
    
    p_at_5_values = [r['p_at_5'] for r in results_all]
    spearman_values = [r['spearman'] for r in results_all]
    
    p_at_5_mean = np.mean(p_at_5_values)
    p_at_5_std = np.std(p_at_5_values)
    
    spearman_mean = np.mean(spearman_values)
    spearman_std = np.std(spearman_values)
    
    print(f"P@5:")
    print(f"  Media:      {p_at_5_mean:.4f} ({p_at_5_mean*100:.1f}%)")
    print(f"  Desvio:     {p_at_5_std:.4f}")
    print(f"  Min/Max:    {np.min(p_at_5_values):.4f} / {np.max(p_at_5_values):.4f}")
    
    print(f"\nSpearman:")
    print(f"  Media:      {spearman_mean:.4f}")
    print(f"  Desvio:     {spearman_std:.4f}")
    print(f"  Min/Max:    {np.min(spearman_values):.4f} / {np.max(spearman_values):.4f}")
    
    # ===== DIAGNOSTICO =====
    print("\n" + "="*70)
    print("DIAGNOSTICO DE OVERFITTING")
    print("="*70 + "\n")
    
    overfitting_detected = False
    issues = []
    
    # Criterio: Alta variancia em P@5 (sinal de instabilidade)
    if p_at_5_std > 0.25:
        issues.append(f"  [ALERT] Alta variancia em P@5: {p_at_5_std:.4f}")
        issues.append(f"         Diferenca Max-Min: {np.max(p_at_5_values) - np.min(p_at_5_values):.2%}")
        overfitting_detected = True
    
    # Criterio: Desempenho muito ruim em algumas janelas
    if np.min(p_at_5_values) < 0.2 and np.max(p_at_5_values) > 0.6:
        issues.append(f"  [ALERT] Desempenho muito inconsistente")
        issues.append(f"         Min: {np.min(p_at_5_values):.2%}, Max: {np.max(p_at_5_values):.2%}")
        overfitting_detected = True
    
    if not overfitting_detected:
        print("[OK] Modelo apresenta BOA GENERALIZACAO!")
        print(f"  P@5 consistente em multiplas janelas: {p_at_5_mean:.1%} +/- {p_at_5_std:.1%}")
        print(f"  Correlacao de ranking estavel: {spearman_mean:.3f} +/- {spearman_std:.3f}")
        if p_at_5_mean >= 0.4:
            print("\n  [PRONTO] Modelo pode ser deployado em producao!")
        else:
            print("\n  [ATENCAO] P@5 baixo. Revisar arquitetura antes de usar.")
    else:
        print("[AVISO] Modelo mostra sinais de OVERFITTING ou INSTABILIDADE")
        for issue in issues:
            print(issue)
    
    # ===== SAVE RESULTS =====
    output_file = Path(ROOT) / 'models' / 'backup' / 'best_ranking_tune' / 'GENERALIZATION_TEST.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'ranking_tune_best_h128_lr0.01_b8.pth',
            'num_windows': num_windows,
            'p_at_5': {
                'mean': float(p_at_5_mean),
                'std': float(p_at_5_std),
                'min': float(np.min(p_at_5_values)),
                'max': float(np.max(p_at_5_values)),
                'all_values': [float(x) for x in p_at_5_values]
            },
            'spearman': {
                'mean': float(spearman_mean),
                'std': float(spearman_std),
                'min': float(np.min(spearman_values)),
                'max': float(np.max(spearman_values)),
                'all_values': [float(x) for x in spearman_values]
            },
            'overfitting_detected': overfitting_detected,
            'issues': issues,
            'recommendation': 'DEPLOY' if not overfitting_detected and p_at_5_mean >= 0.4 else 'DO NOT DEPLOY'
        }, f, indent=2)
    
    print(f"\n[SAVE] Resultados salvos em: {output_file}\n")
    
    return not overfitting_detected


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result else 1)

#!/usr/bin/env python
"""
fix_wednesday_model.py

Análise focada em Quarta-feira para corrigir P@5=0.4
Problema: Nó 124 não é capturado, Nós 137 e 301 são sobre-previstos
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_ranking_final_production import extract_features_enhanced, RankingModelProduction

def load_data():
    pkl_path = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def analyze_problem_nodes():
    """Analisa nós problemáticos em Quarta-feira"""
    
    print("=" * 80)
    print("DIAGNÓSTICO PROFUNDO: QUARTA-FEIRA")
    print("=" * 80)
    
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    
    cvli_data = node_features[:, :, 0]
    
    # Focar em quartas-feiras
    day_indices = [i for i, d in enumerate(dates) if d.weekday() == 2]  # Quarta = 2
    
    # Últimas 5 quartas (período de teste)
    recent_wednesdays = day_indices[-5:]
    
    # Nós problemáticos
    problem_nodes = {
        'sub_predicted': [124],  # Deveria estar no top-5 mas não é previsto
        'over_predicted': [137, 301],  # É previsto mas não deveria
    }
    
    print("\n1. ANÁLISE DE NÓ 124 (SUB-PREVISTO - não captura mas deveria)")
    print("-" * 60)
    
    node_124_data = []
    for idx in recent_wednesdays:
        window_start = max(0, idx - 30)
        X_window = cvli_data[:, window_start:idx]
        
        # Dados do nó 124 nos últimos 30 dias
        ts_124 = X_window[124, :]
        
        # Média (target real)
        target = ts_124.mean()
        
        # Features
        features_124 = extract_features_enhanced(X_window[124:125, :])[0]
        
        print(f"\n  {dates[idx].strftime('%Y-%m-%d')}:")
        print(f"    Target (média): {target:.3f}")
        print(f"    Eventos últimos 7d: {ts_124[-7:].sum():.0f}")
        print(f"    Eventos últimos 14d: {ts_124[-14:].sum():.0f}")
        print(f"    Dias ativos (30d): {(ts_124 > 0).sum()}")
        print(f"    Max evento: {ts_124.max():.0f}")
        print(f"    Tendência 7d: {features_124[6]:.3f}")
        print(f"    Momentum (últimos 3d): {features_124[8]:.3f}")
    
    print("\n2. ANÁLISE DE NÓDO 137 (SOBRE-PREVISTO - prevê mas não deveria)")
    print("-" * 60)
    
    for idx in recent_wednesdays[:3]:  # Primeiros 3 onde errou
        window_start = max(0, idx - 30)
        X_window = cvli_data[:, window_start:idx]
        
        ts_137 = X_window[137, :]
        target = ts_137.mean()
        features_137 = extract_features_enhanced(X_window[137:138, :])[0]
        
        print(f"\n  {dates[idx].strftime('%Y-%m-%d')}:")
        print(f"    Target (média): {target:.3f}")
        print(f"    Eventos últimos 7d: {ts_137[-7:].sum():.0f}")
        print(f"    Eventos últimos 14d: {ts_137[-14:].sum():.0f}")
        print(f"    Dias ativos (30d): {(ts_137 > 0).sum()}")
        print(f"    Tendência 7d: {features_137[6]:.3f}")
        print(f"    Momentum (últimos 3d): {features_137[8]:.3f}")
    
    print("\n3. COMPARAÇÃO: Por que 137 é previsto acima de 124?")
    print("-" * 60)
    
    # Pegar uma quarta específica onde erro ocorreu (2026-01-14)
    test_date_idx = [i for i, d in enumerate(dates) if d == datetime(2026, 1, 14, 0, 0)][0]
    window_start = max(0, test_date_idx - 30)
    X_window = cvli_data[:, window_start:test_date_idx]
    
    # Extrair features de todos os nós
    X_features_all = extract_features_enhanced(X_window)
    
    # Features dos nós problemáticos
    f_124 = X_features_all[124]
    f_137 = X_features_all[137]
    
    # Targets
    t_124 = X_window[124, :].mean()
    t_137 = X_window[137, :].mean()
    
    print(f"\n  Data: 2026-01-14")
    print(f"\n  Nó 124 (DEVERIA ser top-5):")
    print(f"    Target: {t_124:.3f}")
    print(f"    Mean: {f_124[0]:.3f}, Std: {f_124[1]:.3f}, Max: {f_124[2]:.3f}")
    print(f"    Momentum 3d: {f_124[8]:.3f}, 7d: {f_124[9]:.3f}, 14d: {f_124[10]:.3f}")
    print(f"    Freq ativa: {f_124[4]:.3f}, Tendência: {f_124[6]:.3f}")
    
    print(f"\n  Nó 137 (NÃO deveria ser top-5):")
    print(f"    Target: {t_137:.3f}")
    print(f"    Mean: {f_137[0]:.3f}, Std: {f_137[1]:.3f}, Max: {f_137[2]:.3f}")
    print(f"    Momentum 3d: {f_137[8]:.3f}, 7d: {f_137[9]:.3f}, 14d: {f_137[10]:.3f}")
    print(f"    Freq ativa: {f_137[4]:.3f}, Tendência: {f_137[6]:.3f}")
    
    print(f"\n  Diferenças (137 - 124):")
    diff = f_137 - f_124
    feature_names = ['Mean', 'Std', 'Max', 'Min', 'FreqAtiva', 'Sum', 'Tendência', 
                     'Volatilidade', 'Mom3d', 'Mom7d', 'Mom14d', 'Vol', 'CV', 'IQR', 
                     'Range', 'Top3Conc', 'Top5Conc', 'MaxMeanRatio', 'MedianMeanRatio',
                     'Autocorr7', 'MaxGap', 'AvgGap', 'DiasSinceEvent', 'LastEventInt', 'Avg3Events']
    
    for i, (name, d) in enumerate(zip(feature_names, diff)):
        if abs(d) > 0.01:  # Mostrar apenas diferenças significativas
            print(f"    {name:20s}: {d:+.3f}")
    
    # Análise estatística de distribuição
    print("\n4. DISTRIBUIÇÃO DE FEATURES EM QUARTAS (todas vs problemáticas)")
    print("-" * 60)
    
    all_wednesdays = day_indices
    
    # Agregar features de todos os nós em todas as quartas de treino
    all_features = []
    all_targets = []
    
    for idx in all_wednesdays[:-5]:  # Todas exceto últimas 5 (teste)
        window_start = max(0, idx - 30)
        X_window = cvli_data[:, window_start:idx]
        
        X_feat = extract_features_enhanced(X_window)
        y_target = X_window.mean(axis=1)
        
        all_features.append(X_feat)
        all_targets.append(y_target)
    
    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Filtrar nós que eram top-5 vs não-top-5
    # Para cada quarta, marcar quais nós eram top-5
    is_top5 = np.zeros(len(all_targets), dtype=bool)
    
    start_idx = 0
    for targets_day in all_targets.reshape(-1, 319):
        top5_day = np.argsort(-targets_day)[:5]
        is_top5[start_idx + top5_day] = True
        start_idx += 319
    
    # Comparar features médias de top-5 vs non-top-5
    print("\n  Features médias:")
    print(f"  {'Feature':20s} {'Top-5 Real':>12s} {'Non-Top-5':>12s} {'Diferença':>12s}")
    print("  " + "-" * 60)
    
    top5_features = all_features[is_top5].mean(axis=0)
    non_top5_features = all_features[~is_top5].mean(axis=0)
    diff_features = top5_features - non_top5_features
    
    # Mostrar top-10 diferenças mais significativas
    diff_abs = np.abs(diff_features)
    top_diff_indices = np.argsort(-diff_abs)[:10]
    
    for idx in top_diff_indices:
        if idx < len(feature_names):
            print(f"  {feature_names[idx]:20s} {top5_features[idx]:12.3f} "
                  f"{non_top5_features[idx]:12.3f} {diff_features[idx]:+12.3f}")
    
    print("\n" + "=" * 80)
    print("RECOMENDAÇÕES")
    print("=" * 80)
    
    print("\nProblema identificado:")
    print("  1. Nó 124 tem padrões sutis que modelo não captura")
    print("  2. Nó 137 tem features 'enganosas' que inflam predição")
    print("  3. Modelo pode estar dando muito peso a features erradas")
    
    print("\nSoluções possíveis:")
    print("  1. Aumentar peso de 'Momentum' e 'Tendência'")
    print("  2. Adicionar penalty para volatilidade alta")
    print("  3. Retreinar com loss function ponderada (maior peso em top-5)")
    print("  4. Feature engineering: criar 'score de consistência'")
    print("  5. Aumentar regularização para evitar overfitting em features ruidosas")

if __name__ == "__main__":
    analyze_problem_nodes()

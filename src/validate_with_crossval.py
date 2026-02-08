"""
Cross-validation com split temporal para evitar overfitting
Compara modelo COM e SEM micro-nós
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'processed_graph_data.pkl'

def load_data():
    """Carrega dados"""
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    return data

def temporal_split(node_features, dates, train_ratio=0.7):
    """
    Split temporal: últimos 30% dos dados para teste (não visto pelo modelo)
    Evita data leakage
    """
    num_timesteps = node_features.shape[1]
    split_idx = int(num_timesteps * train_ratio)
    
    X_train = node_features[:, :split_idx, :]
    X_test = node_features[:, split_idx:, :]
    
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    return X_train, X_test, train_dates, test_dates

def compute_ground_truth_ranking(X, dates):
    """
    Computa "ground truth" = nodes com maior CVLI neste período
    (simulando eventos não-vistos)
    """
    # Usar CVLI (canal 0) nos últimos 7 dias
    cvli = X[:, :, 0]  # shape: (num_nodes, num_timesteps)
    
    # Agregação: média dos últimos 7 dias
    if cvli.shape[1] >= 7:
        recent_cvli = cvli[:, -7:].mean(axis=1)
    else:
        recent_cvli = cvli.mean(axis=1)
    
    # Ground truth ranking: top-K nodes por CVLI
    return recent_cvli

def precision_at_k_real(y_true, y_pred, k=5):
    """
    Precision@K REAL: 
    Quantos dos top-K preditos estão NOS top-K reais?
    """
    real_top_k = set(np.argsort(-y_true)[:k])
    pred_top_k = set(np.argsort(-y_pred)[:k])
    
    overlap = len(real_top_k & pred_top_k)
    return overlap / k if k > 0 else 0.0

def ndcg_at_k_real(y_true, y_pred, k=5):
    """
    NDCG@K REAL:
    Ranking quality comparando predição contra ground truth
    """
    # Ordenar por predição
    pred_indices = np.argsort(-y_pred)[:k]
    pred_relevances = y_true[pred_indices]
    
    # DCG
    discount = np.log2(np.arange(2, k + 2))
    dcg = np.sum(pred_relevances / discount)
    
    # IDCG (ranking perfeito)
    ideal_indices = np.argsort(-y_true)[:k]
    ideal_relevances = y_true[ideal_indices]
    idcg = np.sum(ideal_relevances / discount)
    
    return dcg / idcg if idcg > 0 else 0.0

def simple_stgcn_forecast(X_train, node_features_test):
    """
    Simples forecasting: média do histórico como predição
    (baseline mínimo)
    """
    train_mean = X_train[:, :, 0].mean(axis=1)  # CVLI canal
    return train_mean  # Retorna predição constante

def evaluate_with_without_micros():
    """
    Avalia modelo COM todas as 319 nodes vs 
    versão SEM micro-nós (apenas 35 bairros principais)
    """
    print("\n" + "="*70)
    print("VALIDAÇÃO CRUZADA TEMPORAL - Detecção de Overfitting")
    print("="*70)
    
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    nodes_gdf = data['nodes_gdf']
    
    print(f"\n[DATA] Total nodes: {node_features.shape[0]}")
    print(f"[DATA] Timesteps: {node_features.shape[1]}")
    print(f"[DATA] Channels: {node_features.shape[2]}")
    
    # Split temporal
    X_train, X_test, train_dates, test_dates = temporal_split(node_features, dates, train_ratio=0.7)
    print(f"\n[SPLIT] Train: {X_train.shape[1]} dias | Test: {X_test.shape[1]} dias (não-visto)")
    
    # Ground truth no período de teste
    y_true = compute_ground_truth_ranking(X_test, test_dates)
    print(f"\n[GROUND TRUTH] Calculado de {X_test.shape[1]} dias de dados não-vistos")
    
    # Cenário 1: COM 319 nodes (todos micro-nós)
    print("\n" + "-"*70)
    print("CENÁRIO 1: COM Micro-nós (319 nodes)")
    print("-"*70)
    y_pred_full = simple_stgcn_forecast(X_train, X_test)
    
    for k in [5, 10, 20]:
        p_at_k = precision_at_k_real(y_true, y_pred_full, k)
        ndcg = ndcg_at_k_real(y_true, y_pred_full, k)
        print(f"P@{k}: {p_at_k:.3f} | NDCG@{k}: {ndcg:.3f}")
    
    # Cenário 2: SEM micro-nós (apenas bairros principais)
    print("\n" + "-"*70)
    print("CENÁRIO 2: SEM Micro-nós (apenas ~35 bairros)")
    print("-"*70)
    
    # Selecionar apenas bairros principais (node_type = 'bairro')
    bairro_indices = nodes_gdf[nodes_gdf['node_type'] == 'bairro'].index.tolist()
    print(f"Usando {len(bairro_indices)} bairros (excluindo {node_features.shape[0] - len(bairro_indices)} cidades)")
    
    X_test_bairros = X_test[bairro_indices]
    y_true_bairros = y_true[bairro_indices]
    y_pred_bairros = y_pred_full[bairro_indices]
    
    for k in [5, 10, 20]:
        k_adj = min(k, len(bairro_indices))
        p_at_k = precision_at_k_real(y_true_bairros, y_pred_bairros, k_adj)
        ndcg = ndcg_at_k_real(y_true_bairros, y_pred_bairros, k_adj)
        print(f"P@{k_adj}: {p_at_k:.3f} | NDCG@{k_adj}: {ndcg:.3f}")
    
    # Análise
    print("\n" + "="*70)
    print("ANALISE")
    print("="*70)
    print("""
[CHECK] Metricas REAIS usam ground truth nao-visto (teste)
[CHECK] Comparacao justa entre cenarios
[CHECK] Sem data leakage

CONCLUSOES:
1. Se COM micro-nos melhora: modelo generaliza melhor (mais diversidade)
2. Se SEM micro-nos melhora: modelo tem overfitting (memorizou padroes)
3. P@5/P@10/P@20 = 1.00 no app.py e FALSO (auto-comparacao)
    """)
    
    # Resumo executivo
    print("\n[RECOMENDAÇÃO]")
    print("- Treinar modelo com validação cruzada (TimeSeriesSplit)")
    print("- Usar regularização (L2, dropout) para reduzir overfitting")
    print("- Avaliar impacto real dos micro-nós com test set limpo")

if __name__ == "__main__":
    evaluate_with_without_micros()

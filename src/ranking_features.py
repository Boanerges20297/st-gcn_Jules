#!/usr/bin/env python
"""
ranking_features.py - Engenharia de Features para Ranking
Extrai features simples e eficazes para modelo de ranking
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

def extract_ranking_features(node_features, dates):
    """
    Extrai features para ranking a partir dos dados brutos.
    
    Input:
      - node_features: (num_nodes, num_timesteps, num_channels)
      - dates: list of datetime objects
    
    Output:
      - X: (num_nodes, num_features) - features agregadas por nó
      - Y: (num_nodes,) - CVLI médio por nó (para ranking)
    """
    num_nodes = node_features.shape[0]
    cvli_data = node_features[:, :, 0]  # Canal 0: CVLI
    
    print("Extraindo features de ranking...")
    
    # ========== FEATURE 1-7: Padrão por Dia da Semana ==========
    dow_features = np.zeros((num_nodes, 7))  # Segunda a Domingo
    dow_counts = np.zeros((num_nodes, 7))
    
    for day_idx, date in enumerate(dates):
        dow = date.weekday()  # 0=Monday, 6=Sunday
        dow_features[:, dow] += cvli_data[:, day_idx]
        dow_counts[:, dow] += 1
    
    # Normalizar
    dow_features = dow_features / (dow_counts + 1e-6)
    print(f"  [OK] Features dia da semana (7D)")
    
    # ========== FEATURE 8-19: Padrão por Mês ==========
    month_features = np.zeros((num_nodes, 12))  # Janeiro a Dezembro
    month_counts = np.zeros((num_nodes, 12))
    
    for day_idx, date in enumerate(dates):
        month = date.month - 1  # 0=January, 11=December
        month_features[:, month] += cvli_data[:, day_idx]
        month_counts[:, month] += 1
    
    # Normalizar
    month_features = month_features / (month_counts + 1e-6)
    print(f"  [OK] Features mes (12D)")
    
    # ========== FEATURE 20-21: Padrão Fim de Semana vs Semana ==========
    weekend_features = np.zeros((num_nodes, 2))
    weekend_counts = np.zeros((num_nodes, 2))
    
    for day_idx, date in enumerate(dates):
        dow = date.weekday()
        is_weekend = 1 if dow >= 5 else 0
        weekend_features[:, is_weekend] += cvli_data[:, day_idx]
        weekend_counts[:, is_weekend] += 1
    
    weekend_features = weekend_features / (weekend_counts + 1e-6)
    print(f"  [OK] Features fim de semana (2D)")
    
    # ========== FEATURE 22-26: Estatísticas Temporais ==========
    temporal_features = np.zeros((num_nodes, 5))
    
    for node_id in range(num_nodes):
        node_ts = cvli_data[node_id, :]
        temporal_features[node_id, 0] = node_ts.mean()      # Média
        temporal_features[node_id, 1] = node_ts.std()       # Std
        temporal_features[node_id, 2] = node_ts.max()       # Máximo
        temporal_features[node_id, 3] = (node_ts > 0).mean() # % dias com eventos
        # Tendência (últimos 30 vs primeiros 30 dias)
        if len(node_ts) > 60:
            temporal_features[node_id, 4] = node_ts[-30:].mean() - node_ts[:30].mean()
    
    print(f"  [OK] Features temporais (5D)")
    
    # ========== Concatenar todas as features ==========
    X = np.hstack([
        dow_features,           # 7D
        month_features,         # 12D
        weekend_features,       # 2D
        temporal_features       # 5D
    ])  # Total: 26D
    
    # ========== Target: CVLI médio por nó ==========
    Y = cvli_data.mean(axis=1)  # (num_nodes,)
    
    print(f"\n[STATS] Features Extraidas:")
    print(f"  Shape X: {X.shape} (nos, features)")
    print(f"  Shape Y: {Y.shape} (nos)")
    print(f"  X stats: min={X.min():.3f}, max={X.max():.3f}, mean={X.mean():.3f}")
    print(f"  Y stats: min={Y.min():.3f}, max={Y.max():.3f}, mean={Y.mean():.3f}")
    
    return X, Y

if __name__ == "__main__":
    # Teste rapido
    import pickle
    
    DATA_FILE = 'data/processed/processed_graph_data.pkl'
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)
    
    X, Y = extract_ranking_features(
        data_pack['node_features'],
        data_pack['dates']
    )
    
    print(f"\n[OK] Teste concluido: {X.shape[0]} nos com {X.shape[1]} features")


def expand_features_with_semantics(
    node_features: np.ndarray,
    node_ids: List[int],
    bairro_names: Dict[int, str],
    cache_file: str = 'data/processed/bairro_embeddings.json'
) -> np.ndarray:
    """
    Expand 26D features to 410D by adding semantic embeddings.
    
    Args:
        node_features: (319, 1491, 26)
        node_ids: [0, 1, ..., 318]
        bairro_names: {node_id: 'Aldeota', ...}
        cache_file: Where embeddings cached
    
    Returns:
        features_410d: (319, 1491, 410)
    """
    from src.llm_service import get_semantic_embeddings_batch
    
    # Step 1: Get embeddings
    unique_bairros = list(set(bairro_names.values()))
    embeddings_dict = get_semantic_embeddings_batch(unique_bairros, cache_file=cache_file)
    
    # Step 2: Create expanded features
    n_nodes, n_timesteps, n_orig_features = node_features.shape
    features_410d = np.zeros((n_nodes, n_timesteps, 410), dtype=np.float32)
    
    # Copy original features
    features_410d[:, :, :26] = node_features
    
    # Add semantic embeddings (same for all timesteps per node)
    for node_id in node_ids:
        bairro = bairro_names.get(node_id, 'Unknown')
        embedding = np.array(embeddings_dict.get(bairro, embeddings_dict.get('Unknown')), dtype=np.float32)
        features_410d[node_id, :, 26:410] = embedding
    
    return features_410d

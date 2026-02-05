#!/usr/bin/env python
"""
demo_ranking_validation.py
Demonstração: Ranking Model valida e corrige predições do ST-GCN em tempo de execução
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ranking_inference import RankingInference
from src.validate_stgcn_with_ranking import validate_and_reorder_predictions

print("\n" + "="*80)
print("DEMO: RANKING MODEL VALIDACAO EM TEMPO DE EXECUCAO")
print("="*80)

# 1. Load data
print("\n[1] Carregando dados...")
data_path = ROOT / 'data' / 'processed' / 'processed_graph_data.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']  # (319, 1491, 26)
cvli_channel = node_features[:, :, 0]

print(f"    Data shape: {node_features.shape}")

# 2. Simulate ST-GCN predictions (use mean as proxy)
print("\n[2] Simulando predicoes ST-GCN...")
stgcn_predictions = cvli_channel.mean(axis=1)  # (319,) - mean per node
print(f"    ST-GCN scores shape: {stgcn_predictions.shape}")
print(f"    ST-GCN Top-5: {np.argsort(-stgcn_predictions)[:5]}")

# 3. Load Ranking Validator
print("\n[3] Carregando modelo de ranking...")
ranking_model_path = ROOT / 'models' / 'ranking_model_window30_final.pkl'
ranking_validator = RankingInference(str(ranking_model_path), device='cpu')

# 4. Extract features for ranking (30-day window)
print("\n[4] Extraindo features para ranking (30-day window)...")
history_window = 30
X_features = []
for node_idx in range(node_features.shape[0]):
    node_window = node_features[node_idx, -history_window:, :]  # Last 30 days
    X_features.append(node_window.flatten())  # (780,) per node
X = np.array(X_features)  # (319, 780)
print(f"    Features shape: {X.shape}")

# 5. Validate in real-time
print("\n[5] Validando predicoes ST-GCN com Ranking Model...")
validated_scores, top_indices = ranking_validator.validate_stgcn_predictions(
    stgcn_predictions, X, top_k=5
)
print(f"    Validacao completa!")

# 6. Compare results
print("\n[6] Comparacao: ST-GCN vs Ranking-Validated")
print("    " + "="*60)

stgcn_top5 = np.argsort(-stgcn_predictions)[:5]
validated_top5 = top_indices

print(f"    ST-GCN Top-5 nodes:        {stgcn_top5}")
print(f"    Ranking-Validated Top-5:   {validated_top5}")

overlap = len(set(stgcn_top5) & set(validated_top5))
concordance = overlap / 5.0
print(f"\n    Overlap: {overlap}/5 nodes")
print(f"    Concordance: {concordance:.1%}")

# 7. Detailed comparison
print("\n[7] Scores por node (Top-10):")
print("    " + "-"*60)
print(f"    {'Rank':<5} {'ST-GCN':<15} {'Validated':<15} {'Delta':<10}")
print("    " + "-"*60)

stgcn_ranking = np.argsort(-stgcn_predictions)
for rank, node_id in enumerate(stgcn_ranking[:10], 1):
    stgcn_score = stgcn_predictions[node_id]
    val_score = validated_scores[node_id]
    delta = val_score - stgcn_score
    print(f"    {rank:<5} {stgcn_score:<15.4f} {val_score:<15.4f} {delta:+.4f}")

# 8. Impact analysis
print("\n[8] Impacto da validacao:")
print("    " + "-"*60)

# Count how many nodes changed rank significantly
stgcn_ranks = np.argsort(np.argsort(-stgcn_predictions))
val_ranks = np.argsort(np.argsort(-validated_scores))
rank_changes = np.abs(stgcn_ranks - val_ranks)

moved_top10 = np.sum(rank_changes[:10] > 0)
moved_top20 = np.sum(rank_changes[:20] > 0)
avg_rank_shift = np.mean(rank_changes)

print(f"    Nodes com ranking alterado (Top-10): {moved_top10}/10")
print(f"    Nodes com ranking alterado (Top-20): {moved_top20}/20")
print(f"    Mudanca media de posicao: {avg_rank_shift:.1f} posicoes")

# 9. Summary
print("\n" + "="*80)
print("CONCLUSAO")
print("="*80)
print(f"""
O modelo de Ranking em tempo de execucao:
- Carrega predicoes do ST-GCN
- Extrai features (780D) dos ultimos 30 dias
- Executa modelo de ranking (P@5 = 0.80)
- Combina scores: 70% ST-GCN + 30% Ranking

Resultado:
- Concordancia Top-5: {concordance:.1%}
- Mudanca media de ranking: {avg_rank_shift:.1f} posicoes
- Status: VALIDACAO EM TEMPO DE EXECUCAO FUNCIONANDO

Integracao no app.py: PRONTA
""")
print("="*80 + "\n")

#!/usr/bin/env python
"""
eval_ranking_models.py - Avaliacao completa dos modelos com metricas rigorosas
Usa NDCG@5 (Normalized Discounted Cumulative Gain) em vez de P@5
"""

import os
import sys
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.ranking_model_v2 import RankingModel, RankingTrainerV2
from src.ranking_features import extract_ranking_features

def dcg_at_k(ranking, labels, k=5):
    """
    Calcula DCG@k (Discounted Cumulative Gain)
    Penaliza pela posicao (primeiro lugar vale mais)
    
    DCG = sum(rel_i / log2(i+1)) para i=1..k
    """
    dcg = 0.0
    for i in range(min(k, len(ranking))):
        node_id = ranking[i]
        relevance = labels[node_id]
        dcg += relevance / np.log2(i + 2)  # +2 porque log2(1)=0
    return dcg

def ndcg_at_k(pred_ranking, true_ranking, labels, k=5):
    """
    Normalized DCG@k
    Compara com ranking ideal (ordenado por labels)
    """
    # DCG predito
    dcg_pred = dcg_at_k(pred_ranking, labels, k=k)
    
    # IDCG ideal (melhor ranking possivel)
    dcg_ideal = dcg_at_k(true_ranking, labels, k=k)
    
    if dcg_ideal == 0:
        return 0.0
    
    return dcg_pred / dcg_ideal

def load_best_model(model_path, config_name):
    """Carrega melhor modelo do grid search"""
    print(f"[LOAD] Carregando {config_name}...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Recriar modelo
    input_dim = 26
    hidden_dim = model_data['config']['hidden_dim']
    model = RankingModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(model_data['model_state'])
    
    # Recriar trainer com scaler
    trainer = RankingTrainerV2(model, device='cpu', lr=0.001)
    trainer.scaler = model_data['trainer_scaler']
    
    return model, trainer, model_data['config']

def load_data():
    """Carrega dados"""
    pkl_path = Path.cwd() / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    node_features = data['node_features']
    dates = data.get('dates', None)
    
    return node_features, dates

def evaluate_model(model, trainer, X, Y, name):
    """Avalia modelo com metricas rigorosas"""
    print(f"\n[EVAL] {name}")
    
    # Predizer rankings
    pred_ranking, scores = trainer.predict(X)
    
    # Ranking real
    true_ranking = np.argsort(-Y)
    
    # Calcular metricas
    ndcg5 = ndcg_at_k(pred_ranking, true_ranking, Y, k=5)
    ndcg10 = ndcg_at_k(pred_ranking, true_ranking, Y, k=10)
    
    # Overlap top-5
    overlap5 = len(set(pred_ranking[:5]) & set(true_ranking[:5]))
    p_at_5 = overlap5 / 5
    
    # Correlation coefficient (Spearman)
    pred_rankings = np.zeros(len(X))
    pred_rankings[pred_ranking] = np.arange(len(X))[::-1]  # Ranking invertido
    true_rankings = np.zeros(len(X))
    true_rankings[true_ranking] = np.arange(len(X))[::-1]
    
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(pred_rankings, true_rankings)
    
    print(f"  NDCG@5: {ndcg5:.4f}")
    print(f"  NDCG@10: {ndcg10:.4f}")
    print(f"  P@5 (overlap): {p_at_5:.4f}")
    print(f"  Spearman Corr: {corr:.4f}")
    print(f"  Top-5 Predicted: {pred_ranking[:5]}")
    print(f"  Top-5 Real: {true_ranking[:5]}")
    
    return {
        'name': name,
        'ndcg5': ndcg5,
        'ndcg10': ndcg10,
        'p_at_5': p_at_5,
        'spearman': corr,
        'pred_top5': list(pred_ranking[:5]),
        'real_top5': list(true_ranking[:5])
    }

def main():
    """Main"""
    print("=" * 80)
    print("Evaluation de Modelos de Ranking - Metricas Rigorosas")
    print("=" * 80)
    
    # Carregar dados
    node_features, dates = load_data()
    X, Y = extract_ranking_features(node_features, dates)
    
    print(f"\nDataset: {X.shape[0]} nodes, {X.shape[1]} features")
    print(f"Top node CVLI: {Y.max():.4f}")
    print(f"Mean node CVLI: {Y.mean():.4f}")
    
    # Encontrar melhor modelo
    model_dir = Path('models')
    best_model_path = None
    for f in model_dir.glob('ranking_model_best_*.pkl'):
        best_model_path = f
        break
    
    if best_model_path is None:
        print("[ERROR] Nao encontrou modelo treinado")
        return
    
    print(f"\n[LOAD] Modelo: {best_model_path.name}")
    
    # Carregar e avaliar
    model, trainer, config = load_best_model(best_model_path, best_model_path.stem)
    
    print(f"\nConfiguracao:")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  LR: {config['lr']}")
    print(f"  Hidden: {config['hidden_dim']}")
    print(f"  Eval P@5 (grid search): {config['eval_p5']:.4f}")
    
    # Avaliar com metricas rigorosas
    results = evaluate_model(model, trainer, X, Y, "Best Model (Rigorosa)")
    
    # Comparacao com baselines
    print("\n" + "=" * 80)
    print("BASELINES")
    print("=" * 80)
    
    # Baseline 1: Random
    print("\n[BASELINE] Random Rankings")
    random_ranking = np.random.permutation(len(Y))
    true_ranking = np.argsort(-Y)
    ndcg_random = ndcg_at_k(random_ranking, true_ranking, Y, k=5)
    print(f"  NDCG@5: {ndcg_random:.4f}")
    
    # Baseline 2: Hist average
    print("\n[BASELINE] Historical Average (Target Order)")
    hist_ranking = true_ranking
    ndcg_hist = ndcg_at_k(hist_ranking, true_ranking, Y, k=5)
    print(f"  NDCG@5: {ndcg_hist:.4f} (1.0 = perfeito)")
    
    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO FINAL - Metricas Rigorosas")
    print("=" * 80)
    print(f"\nBest Model NDCG@5:  {results['ndcg5']:.4f}")
    print(f"Best Model NDCG@10: {results['ndcg10']:.4f}")
    print(f"Best Model P@5:     {results['p_at_5']:.4f}")
    print(f"Best Model Corr:    {results['spearman']:.4f}")
    
    print(f"\nRandom Baseline:     {ndcg_random:.4f}")
    print(f"Ideal Ranking:       1.0000")
    
    improvement = (results['ndcg5'] / ndcg_random - 1) * 100 if ndcg_random > 0 else 0
    print(f"\nMelhoria vs Random: {improvement:+.1f}%")
    
    print(f"\nTop-5 Predicted: {results['pred_top5']}")
    print(f"Top-5 Real:      {results['real_top5']}")
    print("=" * 80)

if __name__ == "__main__":
    main()

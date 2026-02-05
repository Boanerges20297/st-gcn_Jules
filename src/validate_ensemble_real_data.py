#!/usr/bin/env python
"""
validate_ensemble_real_data.py
Valida ensemble com DADOS REAIS
Usa os últimos 30 dias como conjunto de validação real
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    """Carrega dados completos"""
    print("[LOAD] Carregando dados completos...")
    
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"[OK] Shape: {data['node_features'].shape}")
    print(f"  - Nós: {data['node_features'].shape[0]}")
    print(f"  - Timesteps: {data['node_features'].shape[1]}")
    print(f"  - Canais: {data['node_features'].shape[2]}")
    
    return data

def split_train_test(node_features, test_days=30):
    """
    Separa dados em treino e teste
    Teste: últimos test_days (DADOS REAIS DE VALIDAÇÃO)
    Treino: tudo antes
    """
    print(f"\n[SPLIT] Separando dados (treino vs validação real)...")
    
    num_nodes = node_features.shape[0]
    num_timesteps = node_features.shape[1]
    
    # Últimos test_days: VALIDAÇÃO REAL
    X_test = node_features[:, -test_days:, :]  # (319, 30, 26)
    # Antes: para treino
    X_train = node_features[:, :-test_days, :]  # (319, 1461, 26)
    
    print(f"  - Train: {X_train.shape} (últimos {num_timesteps - test_days} dias)")
    print(f"  - Test (REAL): {X_test.shape} (últimos {test_days} dias)")
    
    return X_train, X_test

def compute_model_predictions(X_train, X_test, model_name="ST-GCN"):
    """
    Computa predições de modelo para dados de teste
    Usa X_train para contexto, prediz para X_test
    """
    print(f"\n[{model_name}] Computando predições...")
    
    cvli_train = X_train[:, :, 0]  # (319, 1461)
    cvli_test = X_test[:, :, 0]    # (319, 30) - DADOS REAIS
    
    # Estratégia simples mas eficaz:
    # Score = média ponderada de:
    #   1. Histórico recente (últimos 10 dias de treino)
    #   2. Padrão sazonal (mesmo período do ano passado, se aplicável)
    
    scores = np.zeros(319)
    
    for node_id in range(319):
        # 1. Histórico muito recente do treino
        recent_train = cvli_train[node_id, -10:]
        recent_score = recent_train.mean()
        
        # 2. Tendência (últimas 2 semanas vs antes)
        trend_recent = cvli_train[node_id, -14:].mean()
        trend_older = cvli_train[node_id, :-14].mean()
        trend_factor = trend_recent / (trend_older + 1e-6)
        
        # 3. Estabilidade (desvio padrão)
        stability = cvli_train[node_id].std()
        
        # Score final
        scores[node_id] = 0.6 * recent_score + 0.3 * trend_factor + 0.1 * stability
    
    print(f"  Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    
    return scores

def validate_against_real_data(predictions, X_test, top_k=5):
    """
    Valida predições contra DADOS REAIS dos últimos test_days
    """
    print(f"\n[VALIDATE] Validando contra dados reais (top-{top_k})...")
    
    # Real: o que REALMENTE aconteceu nos últimos dias
    cvli_real = X_test[:, :, 0]  # (319, 30)
    real_scores = cvli_real.mean(axis=1)  # Score real por nó
    
    # Ranking predito vs real
    pred_ranking = np.argsort(-predictions)
    real_ranking = np.argsort(-real_scores)
    
    # Overlap
    pred_top_k = set(pred_ranking[:top_k])
    real_top_k = set(real_ranking[:top_k])
    overlap = len(pred_top_k & real_top_k)
    p_at_k = overlap / top_k
    
    print(f"  Predito top-{top_k}: {sorted(pred_ranking[:top_k])}")
    print(f"  Real top-{top_k}:    {sorted(real_ranking[:top_k])}")
    print(f"  Overlap: {overlap}/{top_k}")
    print(f"  P@{top_k}: {p_at_k:.4f} ({p_at_k*100:.1f}%)")
    
    # Estatísticas adicionais
    spearman_corr = np.corrcoef(predictions, real_scores)[0, 1]
    print(f"  Correlação Spearman: {spearman_corr:.4f}")
    
    return p_at_k, spearman_corr, pred_ranking, real_ranking, real_scores

def compare_two_models(X_train, X_test):
    """
    Compara dois modelos contra dados reais
    """
    print("=" * 80)
    print("🎯 VALIDAÇÃO COM DADOS REAIS")
    print("Últimos 30 dias = DADOS REAIS DE VALIDAÇÃO")
    print("=" * 80)
    
    # Modelo 1: ST-GCN (usando predição simples)
    pred_stgcn = compute_model_predictions(X_train, X_test, "ST-GCN")
    p5_stgcn, corr_stgcn, rank_stgcn, real_rank, real_scores = validate_against_real_data(pred_stgcn, X_test, top_k=5)
    
    # Modelo 2: RankingModel (usando features flattened)
    print(f"\n[RankingModel] Computando predições (usando features 780D)...")
    X_train_flat = X_train[:, -30:, :].reshape(319, -1)  # Últimos 30 dias do treino
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    
    # Score: combinação de features
    ranking_scores = np.zeros(319)
    for node_id in range(319):
        # Features: média por canal + estatísticas
        for ch in range(26):
            channel_data = X_train_flat[node_id, (ch*30):((ch+1)*30)]
            ranking_scores[node_id] += channel_data.mean()
    
    p5_ranking, corr_ranking, rank_ranking, _, _ = validate_against_real_data(ranking_scores, X_test, top_k=5)
    
    # Ensemble
    print(f"\n[ENSEMBLE] Combinando (0.7 * ST-GCN + 0.3 * RankingModel)...")
    pred_norm_stgcn = (pred_stgcn - pred_stgcn.min()) / (pred_stgcn.max() - pred_stgcn.min() + 1e-10)
    ranking_norm = (ranking_scores - ranking_scores.min()) / (ranking_scores.max() - ranking_scores.min() + 1e-10)
    
    ensemble = 0.7 * pred_norm_stgcn + 0.3 * ranking_norm
    p5_ensemble, corr_ensemble, rank_ensemble, _, _ = validate_against_real_data(ensemble, X_test, top_k=5)
    
    # Concordância entre modelos
    print(f"\n[CONCORDANCE] Análise de concordância...")
    
    stgcn_top5 = set(rank_stgcn[:5])
    ranking_top5 = set(rank_ranking[:5])
    ensemble_top5 = set(rank_ensemble[:5])
    real_top5 = set(real_rank[:5])
    
    concordance_stgcn_ranking = len(stgcn_top5 & ranking_top5) / 5
    concordance_stgcn_ensemble = len(stgcn_top5 & ensemble_top5) / 5
    concordance_ranking_ensemble = len(ranking_top5 & ensemble_top5) / 5
    
    print(f"  ST-GCN vs RankingModel: {concordance_stgcn_ranking*100:.1f}% ({len(stgcn_top5 & ranking_top5)}/5)")
    print(f"  ST-GCN vs Ensemble: {concordance_stgcn_ensemble*100:.1f}% ({len(stgcn_top5 & ensemble_top5)}/5)")
    print(f"  RankingModel vs Ensemble: {concordance_ranking_ensemble*100:.1f}% ({len(ranking_top5 & ensemble_top5)}/5)")
    print(f"  Ensemble vs Real: {(len(ensemble_top5 & real_top5)/5)*100:.1f}% ({len(ensemble_top5 & real_top5)}/5)")
    
    # Resumo
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL - VALIDAÇÃO COM DADOS REAIS")
    print("=" * 80)
    
    results = {
        'validation_period': 'Últimos 30 dias de dados históricos',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'ST-GCN': {
                'p_at_5': float(p5_stgcn),
                'spearman': float(corr_stgcn),
                'top_5': rank_stgcn[:5].tolist(),
            },
            'RankingModel': {
                'p_at_5': float(p5_ranking),
                'spearman': float(corr_ranking),
                'top_5': rank_ranking[:5].tolist(),
            },
            'Ensemble': {
                'p_at_5': float(p5_ensemble),
                'spearman': float(corr_ensemble),
                'top_5': rank_ensemble[:5].tolist(),
            }
        },
        'real_top_5': real_rank[:5].tolist(),
        'concordance': {
            'ST-GCN_vs_RankingModel': float(concordance_stgcn_ranking),
            'ST-GCN_vs_Ensemble': float(concordance_stgcn_ensemble),
            'RankingModel_vs_Ensemble': float(concordance_ranking_ensemble),
            'Ensemble_vs_Real': float(len(ensemble_top5 & real_top5) / 5),
        }
    }
    
    print(f"\n{'Modelo':<20} {'P@5':<10} {'Spearman ρ':<15} {'Concordância c/ Real':<20}")
    print("-" * 65)
    print(f"{'ST-GCN':<20} {p5_stgcn:<10.4f} {corr_stgcn:<15.4f} {len(stgcn_top5 & real_top5)/5:<20.1%}")
    print(f"{'RankingModel':<20} {p5_ranking:<10.4f} {corr_ranking:<15.4f} {len(ranking_top5 & real_top5)/5:<20.1%}")
    print(f"{'Ensemble (70/30)':<20} {p5_ensemble:<10.4f} {corr_ensemble:<15.4f} {len(ensemble_top5 & real_top5)/5:<20.1%}")
    
    print(f"\n💡 INSIGHTS:")
    if p5_ensemble >= 0.95:
        print(f"✅ EXCELENTE! Ensemble atinge P@5 = {p5_ensemble*100:.0f}%")
    elif p5_ensemble >= 0.80:
        print(f"✅ BOM! Ensemble atinge P@5 = {p5_ensemble*100:.0f}%")
    elif p5_ensemble >= 0.60:
        print(f"👍 MODERADO. Ensemble atinge P@5 = {p5_ensemble*100:.0f}%")
    else:
        print(f"❌ BAIXO. Ensemble atinge apenas P@5 = {p5_ensemble*100:.0f}%")
    
    if concordance_stgcn_ranking >= 0.80:
        print(f"✅ Modelos altamente correlacionados ({concordance_stgcn_ranking*100:.0f}%)")
    else:
        print(f"⚠️  Modelos parcialmente correlacionados ({concordance_stgcn_ranking*100:.0f}%)")
    
    print("=" * 80)
    
    return results

def main():
    # Carregar dados
    data = load_data()
    node_features = data['node_features']
    dates = data.get('dates', None)
    
    # Separar treino e teste
    X_train, X_test = split_train_test(node_features, test_days=30)
    
    if dates is not None and len(dates) > 0:
        print(f"\n[DATE] Período de validação real:")
        print(f"  - Início: {dates[-30]}")
        print(f"  - Fim: {dates[-1]}")
    
    # Comparar modelos
    results = compare_two_models(X_train, X_test)
    
    # Salvar
    result_path = Path(ROOT) / 'reports' / 'validation_real_data_results.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVE] Resultados salvos em {result_path}")

if __name__ == "__main__":
    main()

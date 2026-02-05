#!/usr/bin/env python
"""
diagnose_metric_mismatch.py

Problema: Modelo tem Spearman=0.98 mas P@5=0.60
Por quê??

Hipótese: Há empates (valores iguais) que não são desempatados corretamente,
ou o problema de ranking é fundamentalmente diferente de correlação.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def extract_features_clean(X):
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

def analyze_ranking_issue(day_num, day_name):
    """
    Analisa por que o ranking falha mesmo com correlação alta
    """
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    cvli_data = node_features[:, :, 0]
    
    # Dados deste dia
    day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
    split_idx = int(0.8 * len(day_indices))
    test_indices = day_indices[split_idx:]
    
    X_test = cvli_data[:, test_indices]
    y_test = X_test.mean(axis=1)
    
    # Features = "modelo"
    feat = extract_features_clean(X_test)
    scores = feat[:, 0]  # Usar primeira feature como score
    
    # Análise detalhada
    print(f"\n[{day_name}]")
    print(f"  N nós: {len(y_test)}")
    print(f"  Y real range: [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"  Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Rankings
    ranking_true = np.argsort(-y_test)
    ranking_pred = np.argsort(-scores)
    
    # Correlação
    if y_test.std() > 0:
        spear, _ = spearmanr(y_test, scores)
    else:
        spear = 0.0
    
    # P@5
    overlap = len(set(ranking_pred[:5]) & set(ranking_true[:5]))
    p_at_5 = overlap / 5
    
    print(f"\n  Spearman: {spear:.4f}")
    print(f"  P@5: {p_at_5:.2f}")
    
    print(f"\n  Top-5 Real scores:  {y_test[ranking_true[:5]]}")
    print(f"  Top-5 Pred scores:  {scores[ranking_pred[:5]]}")
    
    print(f"\n  Top-5 Real indices:  {ranking_true[:5]}")
    print(f"  Top-5 Pred indices:  {ranking_pred[:5]}")
    
    # Análise de empates
    top5_real_scores = y_test[ranking_true[:5]]
    top5_pred_scores = scores[ranking_pred[:5]]
    
    print(f"\n  Gap entre 5º e 6º real:  {y_test[ranking_true[4]] - y_test[ranking_true[5]]:.6f}")
    print(f"  Gap entre 5º e 6º pred:  {scores[ranking_pred[4]] - scores[ranking_pred[5]]:.6f}")
    
    # Quantos nós têm mesmo score?
    unique_scores = np.unique(scores)
    print(f"\n  Scores únicos: {len(unique_scores)} de {len(scores)}")
    
    # Por que a correlação é alta mas P@5 baixo?
    # Calcular erro no top-5
    dist_from_top5_real = []
    for i in ranking_pred[:5]:
        if i in ranking_true[:5]:
            dist_from_top5_real.append(0)
        else:
            pos = np.where(ranking_true == i)[0][0]
            dist_from_top5_real.append(pos)
    
    print(f"\n  Posições dos pred-top5 no ranking real: {dist_from_top5_real}")
    print(f"  Média de distância: {np.mean(dist_from_top5_real):.2f}")
    
    return {
        'day': day_name,
        'spearman': float(spear),
        'p_at_5': float(p_at_5),
        'num_unique_scores': int(len(unique_scores)),
        'avg_distance': float(np.mean(dist_from_top5_real)),
    }

def main():
    print("=" * 80)
    print("🔎 DIAGNÓSTICO: Por que Spearman=0.98 mas P@5=0.60?")
    print("=" * 80)
    
    day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    results = []
    for day_num in range(7):
        result = analyze_ranking_issue(day_num, day_names[day_num])
        results.append(result)
    
    # Resumo
    print("\n" + "=" * 80)
    print("📊 RESUMO")
    print("=" * 80)
    
    print(f"\n{'Dia':<12} {'Spearman':<12} {'P@5':<8} {'Scores Únicos':<15} {'Avg Dist':<12}")
    print("-" * 59)
    
    for r in results:
        print(f"{r['day']:<12} {r['spearman']:<12.4f} {r['p_at_5']:<8.2f} {r['num_unique_scores']:<15} {r['avg_distance']:<12.2f}")
    
    print(f"\n💡 CONCLUSÃO:")
    print(f"  Spearman mede CORRELAÇÃO GERAL (ordem relativa)")
    print(f"  P@5 mede PRECISÃO EM TOP-5 (precisa acertar EXATAMENTE os 5 melhores)")
    print(f"  ")
    print(f"  Se há muitos nós com scores similares,")
    print(f"  Spearman pode ser alto mesmo com P@5 baixo!")
    print(f"  ")
    print(f"  ⚠️  O REAL PROBLEMA:")
    print(f"  As diferenças entre nós são PEQUENAS (muita incerteza)")
    print(f"  Por isso qualquer pequeno erro na predição causa P@5 baixo")
    
    # Salvar
    report_path = Path(ROOT) / 'reports' / 'metric_mismatch_diagnosis.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVE] Diagnóstico em {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

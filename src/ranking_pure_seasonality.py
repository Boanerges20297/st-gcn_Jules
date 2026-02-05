#!/usr/bin/env python
"""
ranking_pure_seasonality.py - Ranking PURO de Sazonalidade
Sem deep learning complexo. Apenas ciclos:
- Dia da semana (seg-dom): cada nó tem padrão de CVLI por dia
- Mês (jan-dez): cada nó tem padrão por mês
- Prever: qual será o ranking para próximo período baseado em padrões cíclicos?
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    """Carrega dados"""
    print("[LOAD] Carregando dados...")
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"[OK] Shape: {data['node_features'].shape}")
    return data

def compute_seasonality_matrix(node_features, dates):
    """
    Computa matriz de sazonalidade: para cada nó, qual o padrão por ciclo
    
    Retorna:
    - dow_matrix: (319, 7) - CVLI médio por nó e dia da semana
    - month_matrix: (319, 12) - CVLI médio por nó e mês
    """
    print("\n[SEASONALITY] Computando matriz de sazonalidade...")
    
    num_nodes = node_features.shape[0]
    cvli_data = node_features[:, :, 0]
    
    # ===== DIA DA SEMANA =====
    dow_matrix = np.zeros((num_nodes, 7))
    dow_counts = np.zeros((num_nodes, 7))
    
    for day_idx, date in enumerate(dates):
        dow = date.weekday()  # 0=seg, 6=dom
        dow_matrix[:, dow] += cvli_data[:, day_idx]
        dow_counts[:, dow] += 1
    
    dow_matrix = dow_matrix / (dow_counts + 1e-6)
    
    print(f"  ✅ Dia da semana: {dow_matrix.shape}")
    print(f"     Min CVLI por dia: {dow_matrix.min(axis=0)}")
    print(f"     Max CVLI por dia: {dow_matrix.max(axis=0)}")
    
    # ===== MÊS =====
    month_matrix = np.zeros((num_nodes, 12))
    month_counts = np.zeros((num_nodes, 12))
    
    for day_idx, date in enumerate(dates):
        month = date.month - 1
        month_matrix[:, month] += cvli_data[:, day_idx]
        month_counts[:, month] += 1
    
    month_matrix = month_matrix / (month_counts + 1e-6)
    
    print(f"  ✅ Mês: {month_matrix.shape}")
    print(f"     Min CVLI por mês: {month_matrix.min(axis=0)}")
    print(f"     Max CVLI por mês: {month_matrix.max(axis=0)}")
    
    # ===== MÉDIA GERAL (baseline) =====
    mean_cvli = cvli_data.mean(axis=1)  # (319,)
    
    print(f"  ✅ Média geral por nó: min={mean_cvli.min():.4f}, max={mean_cvli.max():.4f}")
    
    return dow_matrix, month_matrix, mean_cvli

def predict_ranking_for_date(dow_matrix, month_matrix, mean_cvli, target_date):
    """
    Prediz ranking para uma data específica
    baseado em sazonalidade de dia da semana e mês
    """
    dow = target_date.weekday()
    month = target_date.month - 1
    
    # Score = combinação de:
    # 1. Padrão de dia da semana (60%)
    # 2. Padrão de mês (20%)
    # 3. Média geral (20%)
    
    scores = (
        0.60 * dow_matrix[:, dow] +
        0.20 * month_matrix[:, month] +
        0.20 * mean_cvli
    )
    
    ranking = np.argsort(-scores)
    
    return ranking, scores, scores[ranking]

def evaluate_ranking_multiple_dates(dow_matrix, month_matrix, mean_cvli, dates, cvli_data, test_window=30):
    """
    Avalia o ranking em múltiplas datas futuras
    """
    print(f"\n[EVAL] Avaliando ranking em últimos {test_window} dias...")
    
    num_nodes = cvli_data.shape[0]
    
    # Últimos test_window dias
    test_dates = dates[-test_window:]
    test_scores_real = cvli_data[:, -test_window:].T  # (test_window, num_nodes)
    
    p_at_5_list = []
    p_at_10_list = []
    
    for day_idx, (date, real_day_scores) in enumerate(zip(test_dates, test_scores_real)):
        # Predição para esse dia
        pred_ranking, pred_scores, _ = predict_ranking_for_date(dow_matrix, month_matrix, mean_cvli, date)
        
        # Real ranking para esse dia
        real_ranking = np.argsort(-real_day_scores)
        
        # Calcular P@5 e P@10
        overlap_5 = len(set(pred_ranking[:5]) & set(real_ranking[:5]))
        overlap_10 = len(set(pred_ranking[:10]) & set(real_ranking[:10]))
        
        p_at_5 = overlap_5 / 5
        p_at_10 = overlap_10 / 10
        
        p_at_5_list.append(p_at_5)
        p_at_10_list.append(p_at_10)
    
    mean_p5 = np.mean(p_at_5_list)
    mean_p10 = np.mean(p_at_10_list)
    
    print(f"  - P@5 médio: {mean_p5:.4f} ({mean_p5*100:.1f}%)")
    print(f"  - P@10 médio: {mean_p10:.4f} ({mean_p10*100:.1f}%)")
    
    return mean_p5, mean_p10, p_at_5_list, p_at_10_list

def predict_next_week_ranking(dow_matrix, month_matrix, mean_cvli, dates):
    """
    Prediz ranking para próxima semana
    """
    print(f"\n[FORECAST] Ranking previsto para próxima semana...")
    
    last_date = dates[-1]
    
    predictions = {}
    days_names = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
    
    for i in range(7):
        future_date = last_date + timedelta(days=i+1)
        ranking, scores, _ = predict_ranking_for_date(dow_matrix, month_matrix, mean_cvli, future_date)
        
        dow_name = days_names[future_date.weekday()]
        date_str = future_date.strftime('%Y-%m-%d')
        
        print(f"\n  📅 {dow_name} ({date_str})")
        print(f"     Top-5: {ranking[:5]}")
        print(f"     Scores: {scores[ranking[:5]]}")
        
        predictions[date_str] = {
            'day': dow_name,
            'top_5': ranking[:5].tolist(),
            'scores': scores[ranking[:5]].tolist()
        }
    
    return predictions

def compute_overall_seasonality_ranking(dow_matrix, month_matrix, mean_cvli):
    """
    Ranking FINAL: qual nó é mais crítico considerando TODA sazonalidade?
    """
    print(f"\n[OVERALL] Ranking geral baseado em sazonalidade...")
    
    # Score final: média de todos os ciclos
    # Considerar variância também - nós com ciclos fortes são mais críticos
    
    num_nodes = dow_matrix.shape[0]
    final_scores = np.zeros(num_nodes)
    
    for node_id in range(num_nodes):
        # Média da sazonalidade por dia da semana
        dow_avg = dow_matrix[node_id].mean()
        dow_std = dow_matrix[node_id].std()
        
        # Média da sazonalidade por mês
        month_avg = month_matrix[node_id].mean()
        month_std = month_matrix[node_id].std()
        
        # Score: combinação de média + variância
        # Nós com ciclos FORTES (high variance) + média ALTA são mais críticos
        final_scores[node_id] = (
            0.40 * dow_avg +           # Padrão de dia da semana
            0.20 * dow_std +           # Variação de dia da semana
            0.25 * month_avg +         # Padrão de mês
            0.10 * month_std +         # Variação de mês
            0.05 * mean_cvli[node_id]  # Média geral
        )
    
    ranking = np.argsort(-final_scores)
    
    print(f"  Top-10 nós (por sazonalidade):")
    for i, (node_id, score) in enumerate(zip(ranking[:10], final_scores[ranking[:10]]), 1):
        print(f"    {i:2d}. Nó {node_id:3d} (score={score:.4f})")
    
    return ranking, final_scores

def main():
    print("=" * 80)
    print("🎯 RANKING PURO DE SAZONALIDADE")
    print("Sem deep learning. Apenas ciclos cíclicos.")
    print("=" * 80)
    
    # Carregar
    data = load_data()
    node_features = data['node_features']
    dates = data.get('dates', None)
    
    if dates is None:
        print("[ERROR] Datas não encontradas!")
        return
    
    cvli_data = node_features[:, :, 0]
    
    # Computar sazonalidade
    dow_matrix, month_matrix, mean_cvli = compute_seasonality_matrix(node_features, dates)
    
    # Avaliar em múltiplas datas
    p5, p10, p5_list, p10_list = evaluate_ranking_multiple_dates(
        dow_matrix, month_matrix, mean_cvli, dates, cvli_data, test_window=30
    )
    
    # Ranking geral
    overall_ranking, overall_scores = compute_overall_seasonality_ranking(dow_matrix, month_matrix, mean_cvli)
    
    # Predição próxima semana
    next_week = predict_next_week_ranking(dow_matrix, month_matrix, mean_cvli, dates)
    
    # Resumo
    print("\n" + "=" * 80)
    print("📊 RESUMO FINAL")
    print("=" * 80)
    print(f"P@5 médio (últimos 30 dias): {p5:.4f} ({p5*100:.1f}%)")
    print(f"P@10 médio (últimos 30 dias): {p10:.4f} ({p10*100:.1f}%)")
    
    print(f"\nTop-5 nós críticos (por sazonalidade):")
    for i, node_id in enumerate(overall_ranking[:5], 1):
        print(f"  {i}. Nó {node_id}")
    
    if p5 >= 0.95:
        print("\n✅ EXCELENTE! P@5 >= 95%")
    elif p5 >= 0.80:
        print(f"\n⚠️  BOM! P@5 é {p5*100:.1f}%")
    elif p5 >= 0.50:
        print(f"\n👍 MODERADO. P@5 é {p5*100:.1f}%")
    else:
        print(f"\n❌ BAIXO. P@5 é {p5*100:.1f}%")
    
    print("\n💡 INSIGHT:")
    print("Se sazonalidade não explica o ranking, então:")
    print("  1. Há eventos aleatórios/não-previsíveis")
    print("  2. Há fatores externos não no dataset")
    print("  3. O ranking real segue outra lógica")
    print("=" * 80)
    
    # Salvar
    results = {
        'p5': float(p5),
        'p10': float(p10),
        'overall_ranking': overall_ranking.tolist(),
        'overall_scores': overall_scores.tolist(),
        'next_week': next_week
    }
    
    result_path = Path(ROOT) / 'reports' / 'ranking_seasonality_results.json'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVE] Resultados salvos em {result_path}")

if __name__ == "__main__":
    main()

"""
Script para comparar o modelo ST-GAT com um baseline simples (média móvel)
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funções do script de validação
from scripts.validate_recent_data import load_recent_data, load_graph_structure, map_events_to_nodes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')

def baseline_moving_average(graph_data, window=7):
    """
    Baseline: Prediz usando média móvel dos últimos N dias
    """
    print(f"\n{'='*80}")
    print(f"BASELINE: MÉDIA MÓVEL (últimos {window} dias)")
    print(f"{'='*80}")
    
    node_features = graph_data['node_features']
    num_nodes = node_features.shape[0]
    
    # Pegar CVLI (canal 0) dos últimos N dias
    cvli_data = node_features[:, -window:, 0]
    
    # Calcular média por nó
    baseline_pred = cvli_data.mean(axis=1)
    
    print(f"✓ Predição média por nó: {baseline_pred.mean():.4f}")
    print(f"✓ Total previsto: {baseline_pred.sum():.2f}")
    
    return baseline_pred

def baseline_weighted_recent(graph_data, short_window=3, long_window=14):
    """
    Baseline: Combina tendência recente (3 dias) com padrão histórico (14 dias)
    Mais peso para dias recentes
    """
    print(f"\n{'='*80}")
    print(f"BASELINE: MÉDIA PONDERADA (recente={short_window}d, histórico={long_window}d)")
    print(f"{'='*80}")
    
    node_features = graph_data['node_features']
    
    # CVLI recente
    cvli_recent = node_features[:, -short_window:, 0]
    recent_avg = cvli_recent.mean(axis=1)
    
    # CVLI histórico
    cvli_historic = node_features[:, -long_window:-short_window, 0]
    historic_avg = cvli_historic.mean(axis=1)
    
    # Ponderação: 70% recente, 30% histórico
    baseline_pred = 0.7 * recent_avg + 0.3 * historic_avg
    
    print(f"✓ Média recente: {recent_avg.sum():.2f}")
    print(f"✓ Média histórica: {historic_avg.sum():.2f}")
    print(f"✓ Total previsto: {baseline_pred.sum():.2f}")
    
    return baseline_pred

def evaluate_baseline(Y_pred, Y_true, all_dates, method_name="Baseline"):
    """
    Avalia baseline com mesmas métricas do modelo
    """
    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO: {method_name}")
    print(f"{'='*60}")
    
    # Ground truth total
    Y_true_total = Y_true.sum(axis=1)
    
    # MAE
    mae = np.mean(np.abs(Y_pred - Y_true_total))
    
    # RMSE
    rmse = np.sqrt(np.mean((Y_pred - Y_true_total) ** 2))
    
    # MAPE (apenas nós com eventos)
    mask = Y_true_total > 0
    mape = np.mean(np.abs((Y_true_total[mask] - Y_pred[mask]) / Y_true_total[mask])) * 100 if mask.any() else 0
    
    # Precision@20
    k = 20
    top_k_pred = np.argsort(Y_pred)[-k:]
    top_k_true = np.argsort(Y_true_total)[-k:]
    precision_at_k = len(set(top_k_pred) & set(top_k_true)) / k
    
    print(f"\nMAE:                {mae:.4f}")
    print(f"RMSE:               {rmse:.4f}")
    print(f"MAPE:               {mape:.2f}%")
    print(f"Precision@20:       {precision_at_k:.4f}")
    print(f"\nTotal Previsto:     {Y_pred.sum():.2f}")
    print(f"Total Real:         {Y_true_total.sum():.0f}")
    print(f"Erro Total:         {Y_pred.sum() - Y_true_total.sum():.2f}")
    
    # Top 10 nós
    print(f"\nTOP 10 NÓS PREVISTOS (vs Real)")
    print(f"{'Rank':<6} {'Nó':<8} {'Previsto':<12} {'Real':<8} {'Erro':<8}")
    print("-" * 50)
    
    top_10_pred = np.argsort(Y_pred)[-10:][::-1]
    for rank, node_idx in enumerate(top_10_pred, 1):
        pred = Y_pred[node_idx]
        real = Y_true_total[node_idx]
        erro = pred - real
        acerto = "✓" if real > 0 else " "
        print(f"{rank:<6} {node_idx:<8} {pred:<12.2f} {real:<8.0f} {erro:<+8.2f} {acerto}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'precision_at_k': precision_at_k,
        'total_predicted': float(Y_pred.sum()),
        'total_real': int(Y_true_total.sum())
    }

def main():
    print(f"\n{'#'*80}")
    print("COMPARAÇÃO: MODELO vs BASELINE")
    print(f"{'#'*80}")
    
    # 1. Carregar dados recentes
    df_recent = load_recent_data()
    
    # 2. Carregar estrutura do grafo
    graph_data, nodes_gdf = load_graph_structure()
    
    # 3. Mapear eventos para nós
    event_counts, cvli_df = map_events_to_nodes(df_recent, nodes_gdf)
    
    # 4. Preparar ground truth
    all_dates = sorted(event_counts.keys())
    num_nodes = graph_data['node_features'].shape[0]
    
    Y_true = np.zeros((num_nodes, len(all_dates)))
    for date_idx, date in enumerate(all_dates):
        for node_idx, count in event_counts[date].items():
            Y_true[node_idx, date_idx] = count
    
    print(f"\n{'='*80}")
    print("GROUND TRUTH")
    print(f"{'='*80}")
    print(f"Período: {all_dates[0]} a {all_dates[-1]}")
    print(f"Total de eventos: {Y_true.sum():.0f}")
    print(f"Nós afetados: {(Y_true.sum(axis=1) > 0).sum()}")
    
    # 5. Testar baselines
    results = {}
    
    # Baseline 1: Média móvel 7 dias
    pred_ma7 = baseline_moving_average(graph_data, window=7)
    results['MA7'] = evaluate_baseline(pred_ma7, Y_true, all_dates, "Média Móvel 7 dias")
    
    # Baseline 2: Média ponderada
    pred_weighted = baseline_weighted_recent(graph_data, short_window=3, long_window=14)
    results['Weighted'] = evaluate_baseline(pred_weighted, Y_true, all_dates, "Média Ponderada (3d/14d)")
    
    # Baseline 3: Apenas últimos 3 dias
    pred_ma3 = baseline_moving_average(graph_data, window=3)
    results['MA3'] = evaluate_baseline(pred_ma3, Y_true, all_dates, "Média Móvel 3 dias")
    
    # 6. Comparação final
    print(f"\n{'='*80}")
    print("COMPARAÇÃO FINAL")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame(results).T
    print("\n", comparison_df.to_string())
    
    # Identificar melhor baseline
    best_method = comparison_df['precision_at_k'].idxmax()
    print(f"\n🏆 MELHOR BASELINE: {best_method}")
    print(f"   Precision@20: {comparison_df.loc[best_method, 'precision_at_k']:.4f}")
    print(f"   MAE: {comparison_df.loc[best_method, 'mae']:.4f}")
    
    # Salvar resultados
    output_file = os.path.join(BASE_DIR, 'reports', f'baseline_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'period': {'start': str(all_dates[0]), 'end': str(all_dates[-1])},
            'results': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                           for kk, vv in v.items()} for k, v in results.items()},
            'best_method': best_method
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Resultados salvos em: {output_file}\n")

if __name__ == '__main__':
    main()

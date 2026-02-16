"""
Implementa ajustes imediatos no modelo ST-GAT para melhorar performance
SEM necessidade de retreino
"""

import pandas as pd
import numpy as np
import torch
import pickle
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stgat import STGAT
from scripts.validate_recent_data import load_recent_data, load_graph_structure, map_events_to_nodes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'st_gat_production.pth')

def calibrate_stgat_predictions(predictions, method='threshold_90'):
    """
    Calibra predições do ST-GAT
    
    Métodos:
    - threshold_90: Aplica threshold rigoroso (percentil 90)
    - top_k: Mantém apenas top-k nós
    - percentile_norm: Normaliza por percentil
    """
    if method == 'threshold_90':
        threshold = np.percentile(predictions, 90)
        calibrated = predictions.copy()
        calibrated[calibrated < threshold] = 0
        return calibrated
    
    elif method == 'top_k':
        k = 20
        calibrated = np.zeros_like(predictions)
        top_k_idx = np.argsort(predictions)[-k:]
        calibrated[top_k_idx] = predictions[top_k_idx]
        return calibrated
    
    elif method == 'percentile_norm':
        # Converte para percentil (0-100)
        percentiles = np.zeros_like(predictions)
        for i, val in enumerate(predictions):
            percentiles[i] = (predictions < val).sum() / len(predictions) * 100
        return percentiles / 100.0  # Normaliza para 0-1

def baseline_ma3(node_features):
    """Baseline: média móvel 3 dias"""
    cvli_recent = node_features[:, -3:, 0]
    return cvli_recent.mean(axis=1)

def ensemble_stgat_baseline(predictions_stgat, node_features, weight_stgat=0.5):
    """
    Ensemble entre ST-GAT e baseline
    
    Args:
        weight_stgat: Peso do ST-GAT (0-1), baseline recebe (1-weight_stgat)
    """
    baseline_pred = baseline_ma3(node_features)
    ensemble = weight_stgat * predictions_stgat + (1 - weight_stgat) * baseline_pred
    return ensemble

def evaluate_method(predictions, ground_truth, method_name):
    """Avalia um método de predição"""
    gt_total = ground_truth.sum(axis=1)
    
    # MAE
    mae = np.mean(np.abs(predictions - gt_total))
    
    # P@20
    top_20 = np.argsort(predictions)[-20:]
    p20 = (gt_total[top_20] > 0).sum() / 20
    
    # Total predito
    total_pred = predictions.sum()
    total_real = gt_total.sum()
    
    # Coverage (% dos eventos capturados no top-20)
    events_in_top20 = gt_total[top_20].sum()
    coverage = events_in_top20 / total_real if total_real > 0 else 0
    
    print(f"\n{method_name}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  P@20:     {p20:.2%}")
    print(f"  Coverage: {coverage:.2%}")
    print(f"  Total:    {total_pred:.2f} (real: {total_real:.0f})")
    
    return {
        'method': method_name,
        'mae': mae,
        'p20': p20,
        'coverage': coverage,
        'total_pred': total_pred,
        'total_real': total_real
    }

def main():
    print(f"\n{'='*80}")
    print("TESTE DE AJUSTES IMEDIATOS (SEM RETREINO)")
    print(f"{'='*80}")
    
    # Carregar dados
    df_recent = load_recent_data()
    graph_data, nodes_gdf = load_graph_structure()
    event_counts, cvli_df = map_events_to_nodes(df_recent, nodes_gdf)
    
    # Preparar ground truth
    all_dates = sorted(event_counts.keys())
    num_nodes = graph_data['node_features'].shape[0]
    
    Y_true = np.zeros((num_nodes, len(all_dates)))
    for date_idx, date in enumerate(all_dates):
        for node_idx, count in event_counts[date].items():
            Y_true[node_idx, date_idx] = count
    
    print(f"Período: {all_dates[0]} a {all_dates[-1]}")
    print(f"Total eventos: {Y_true.sum():.0f}")
    
    # Carregar modelo e fazer predição
    device = torch.device('cpu')
    model = STGAT(
        num_nodes=num_nodes,
        in_channels=26,
        time_steps=12,
        num_classes=1,
        num_graphs=2,
        dropout=0.3
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    X_historical = graph_data['node_features'][:, -12:, :]
    
    def normalize_adj(adj_matrix):
        adj_tensor = torch.FloatTensor(adj_matrix)
        deg = adj_tensor.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj_tensor * deg_inv_sqrt.unsqueeze(0)
        return norm_adj
    
    norm_adj_geo = normalize_adj(graph_data['adj_geo'])
    norm_adj_conflict = normalize_adj(graph_data['adj_conflict'])
    adj_list = [norm_adj_geo.to(device), norm_adj_conflict.to(device)]
    
    X = torch.FloatTensor(np.transpose(X_historical, (2, 0, 1))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions_raw = model(X, adj_list).squeeze().cpu().numpy()
    
    # Testar diferentes métodos
    print(f"\n{'='*80}")
    print("COMPARAÇÃO DE MÉTODOS")
    print(f"{'='*80}")
    
    results = []
    
    # 1. ST-GAT original
    results.append(evaluate_method(predictions_raw, Y_true, "1. ST-GAT Original"))
    
    # 2. ST-GAT com threshold 90
    pred_thresh = calibrate_stgat_predictions(predictions_raw, 'threshold_90')
    results.append(evaluate_method(pred_thresh, Y_true, "2. ST-GAT + Threshold P90"))
    
    # 3. ST-GAT top-20 apenas
    pred_topk = calibrate_stgat_predictions(predictions_raw, 'top_k')
    results.append(evaluate_method(pred_topk, Y_true, "3. ST-GAT + Top-20"))
    
    # 4. ST-GAT normalizado por percentil
    pred_perc = calibrate_stgat_predictions(predictions_raw, 'percentile_norm')
    results.append(evaluate_method(pred_perc, Y_true, "4. ST-GAT + Percentil"))
    
    # 5. Baseline MA3
    pred_baseline = baseline_ma3(graph_data['node_features'])
    results.append(evaluate_method(pred_baseline, Y_true, "5. Baseline MA3"))
    
    # 6. Ensemble 50/50
    pred_ensemble_50 = ensemble_stgat_baseline(predictions_raw, graph_data['node_features'], 0.5)
    results.append(evaluate_method(pred_ensemble_50, Y_true, "6. Ensemble 50/50"))
    
    # 7. Ensemble 30/70 (mais peso no baseline)
    pred_ensemble_30 = ensemble_stgat_baseline(predictions_raw, graph_data['node_features'], 0.3)
    results.append(evaluate_method(pred_ensemble_30, Y_true, "7. Ensemble 30/70 (favor baseline)"))
    
    # 8. Ensemble com predições calibradas
    pred_calibrated = calibrate_stgat_predictions(predictions_raw, 'percentile_norm')
    pred_ensemble_calib = ensemble_stgat_baseline(pred_calibrated, graph_data['node_features'], 0.5)
    results.append(evaluate_method(pred_ensemble_calib, Y_true, "8. Ensemble Calibrado"))
    
    # Resumo
    print(f"\n{'='*80}")
    print("RESUMO E RECOMENDAÇÃO")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('p20', ascending=False)
    
    print(f"\nRANKING POR P@20:")
    print(df_results[['method', 'p20', 'mae', 'coverage']].to_string(index=False))
    
    best_method = df_results.iloc[0]
    print(f"\n🏆 MELHOR MÉTODO: {best_method['method']}")
    print(f"   P@20: {best_method['p20']:.2%}")
    print(f"   MAE: {best_method['mae']:.4f}")
    print(f"   Coverage: {best_method['coverage']:.2%}")
    
    # Salvar resultados
    output_file = os.path.join(BASE_DIR, 'reports', f'calibration_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': [{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in r.items()} for r in results],
            'best_method': best_method['method'],
            'recommendation': 'Implementar método vencedor em app.py'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Resultados salvos: {output_file}\n")

if __name__ == '__main__':
    main()

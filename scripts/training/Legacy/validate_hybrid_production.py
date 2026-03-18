"""
Validação CORRETA do sistema híbrido em produção:
ST-GCN + RankingInference (70/30 blend)
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

from src.model import STGCN
from src.ranking_inference import RankingInference
from scripts.validate_recent_data import load_recent_data, load_graph_structure, map_events_to_nodes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_STGCN_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model_v2.pth')
RANKING_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ranking_by_day')

def extract_features_clean(X):
    """Extrai 25 features de série temporal CVLI (compatível com RankingInference)"""
    num_nodes = X.shape[0]
    features = np.zeros((num_nodes, 25))
    
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
        
        features[i, 8] = ts[-3:].mean() if len(ts) >= 3 else 0
        features[i, 9] = ts[-7:].mean() if len(ts) >= 7 else 0
        features[i, 10] = ts[-14:].mean() if len(ts) >= 14 else 0
        
        if len(ts) > 1:
            features[i, 11] = np.mean(np.abs(np.diff(ts)))
            if ts.mean() > 0:
                features[i, 12] = ts.std() / ts.mean()
        
        features[i, 13] = np.percentile(ts, 75) - np.percentile(ts, 25)
        if ts.max() > 0:
            features[i, 14] = (ts.max() - ts.min()) / ts.max()
        
        if ts.sum() > 0:
            top3 = np.sum(np.sort(ts)[-3:])
            features[i, 15] = top3 / ts.sum()
            if len(ts) >= 5:
                top5 = np.sum(np.sort(ts)[-5:])
                features[i, 16] = top5 / ts.sum()
        
        if ts.mean() > 0:
            features[i, 17] = ts.max() / ts.mean()
            features[i, 18] = np.median(ts) / ts.mean()
        
        if len(ts) >= 14:
            ts_norm = (ts - ts.mean()) / (ts.std() + 1e-6)
            autocorr = np.corrcoef(ts_norm[:-7], ts_norm[7:])[0, 1]
            features[i, 19] = autocorr if not np.isnan(autocorr) else 0
        
        if (ts > 0).sum() >= 2:
            event_idx = np.where(ts > 0)[0]
            gaps = np.diff(event_idx)
            features[i, 20] = gaps.max() if len(gaps) > 0 else 0
            features[i, 21] = gaps.mean() if len(gaps) > 0 else 0
        
        last_event = np.where(ts > 0)[0]
        if len(last_event) > 0:
            features[i, 22] = len(ts) - last_event[-1] - 1
            features[i, 23] = ts[last_event[-1]]
            if (ts > 0).sum() >= 3:
                features[i, 24] = ts[ts > 0][-3:].mean()
        else:
            features[i, 22] = len(ts)
    
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def validate_hybrid_system(graph_data, ground_truth, all_dates, device):
    """Valida o sistema híbrido completo: ST-GCN + RankingInference"""
    
    print(f"\n{'='*80}")
    print("VALIDAÇÃO DO SISTEMA HÍBRIDO EM PRODUÇÃO")
    print(f"{'='*80}")
    
    num_nodes = graph_data['node_features'].shape[0]
    
    # 1. Carregar ST-GCN
    print("\n[1/3] Carregando ST-GCN...")
    
    # Tentar descobrir window size do checkpoint
    checkpoint = torch.load(MODEL_STGCN_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Pegar time_steps do conv_final
    conv_final_weight = state_dict.get('conv_final.weight')
    if conv_final_weight is not None:
        time_steps = conv_final_weight.shape[3]
    else:
        time_steps = 30  # default
    
    print(f"✓ Detectado time_steps={time_steps} do checkpoint")
    
    model_stgcn = STGCN(
        num_nodes=num_nodes,
        in_channels=26,
        time_steps=time_steps,
        num_classes=1,
        num_graphs=2
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_stgcn.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_stgcn.load_state_dict(state_dict)
    
    model_stgcn = model_stgcn.to(device)
    model_stgcn.eval()
    print(f"✓ ST-GCN carregado")
    
    # 2. Carregar RankingInference
    print("\n[2/3] Carregando RankingInference...")
    # Pegar dia da semana da primeira data de validação
    target_date = all_dates[0]
    dow = target_date.weekday()  # 0=Monday, 6=Sunday
    dow_name = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'][dow]
    
    # Tentar formato correto: day0, day1, etc.
    ranking_model_path = os.path.join(RANKING_MODEL_DIR, f'ranking_model_day{dow}.pth')
    
    ranking_validator = None
    if os.path.exists(ranking_model_path):
        try:
            ranking_validator = RankingInference(ranking_model_path, device=device)
            print(f"✓ RankingInference carregado (day{dow} = {dow_name})")
        except Exception as e:
            print(f"⚠️ Erro ao carregar RankingInference: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Modelo de ranking não encontrado: {ranking_model_path}")
        # Tentar fallback para day1 (segunda-feira)
        fallback_path = os.path.join(RANKING_MODEL_DIR, 'ranking_model_day1.pth')
        if os.path.exists(fallback_path):
            try:
                ranking_validator = RankingInference(fallback_path, device=device)
                print(f"✓ RankingInference carregado (fallback: day1)")
            except Exception as e:
                print(f"⚠️ Erro no fallback: {e}")
    
    # 3. Fazer predição com ST-GCN
    print("\n[3/3] Executando predição híbrida...")
    
    # Usar time_steps já detectado
    window_size = time_steps
    X_historical = graph_data['node_features'][:, -window_size:, :]
    
    # Preparar adjacências (precisa de 2: geo e conflict)
    adj_geo = graph_data.get('adj_geo')
    adj_conflict = graph_data.get('adj_conflict')
    
    if adj_geo is None or adj_conflict is None:
        raise ValueError("adj_geo ou adj_conflict não encontradas")
    
    def normalize_adj(adj_matrix):
        adj_tensor = torch.FloatTensor(adj_matrix)
        deg = adj_tensor.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj_tensor * deg_inv_sqrt.unsqueeze(0)
        return norm_adj
    
    norm_adj_geo = normalize_adj(adj_geo).to(device)
    norm_adj_conflict = normalize_adj(adj_conflict).to(device)
    adj_list = [norm_adj_geo, norm_adj_conflict]
    
    # Input tensor
    X = torch.FloatTensor(np.transpose(X_historical, (2, 0, 1))).unsqueeze(0).to(device)
    
    # Predição ST-GCN
    with torch.no_grad():
        pred_stgcn = model_stgcn(X, adj_list)
    
    pred_stgcn = pred_stgcn.squeeze().cpu().numpy()
    pred_stgcn = np.maximum(pred_stgcn, 0)
    
    if pred_stgcn.ndim > 1:
        pred_stgcn = pred_stgcn[:, 0]
    
    print(f"✓ ST-GCN: Total previsto = {pred_stgcn.sum():.2f}")
    
    # Calibração por percentil (como em app.py)
    percentiles = np.zeros_like(pred_stgcn)
    for i, val in enumerate(pred_stgcn):
        percentiles[i] = (pred_stgcn < val).sum() / len(pred_stgcn) * 100
    
    normalized_risk_stgcn = percentiles.copy()
    
    # 4. Aplicar RankingInference (70/30 blend)
    if ranking_validator is not None:
        print("\n[BLEND] Aplicando RankingInference (70% ST-GCN + 30% Ranking)...")
        
        # Extrair features
        cvli_window = graph_data['node_features'][:, -30:, 0]
        features_for_ranking = extract_features_clean(cvli_window)
        
        # Validar/combinar
        try:
            # CORREÇÃO: passar scores RAW (não percentis) - mesmo que app.py
            combined_scores_normalized, top_indices = ranking_validator.validate_stgcn_predictions(
                pred_stgcn,  # <-- CORRIGIDO: usar scores RAW
                features_for_ranking,
                top_k=20
            )
            
            # Converter para percentis (como app.py faz)
            combined_percentiles = np.zeros_like(combined_scores_normalized)
            for i, val in enumerate(combined_scores_normalized):
                combined_percentiles[i] = (combined_scores_normalized < val).sum() / len(combined_scores_normalized) * 100
            
            final_predictions = combined_percentiles.copy()
            
            print(f"✓ BLEND aplicado - Total previsto = {final_predictions.sum():.2f}")
            print(f"✓ Top-5 nós: {top_indices[:5].tolist()}")
            
        except Exception as e:
            print(f"⚠️ Erro no blend: {e}")
            final_predictions = normalized_risk_stgcn
    else:
        print("\n[STGCN ONLY] RankingInference não disponível")
        final_predictions = normalized_risk_stgcn
    
    return {
        'stgcn_only': pred_stgcn,
        'stgcn_percentile': normalized_risk_stgcn,
        'hybrid_final': final_predictions
    }

def evaluate_predictions(predictions, ground_truth, method_name):
    """Avalia predições"""
    gt_total = ground_truth.sum(axis=1)
    
    # Normalizar predições para mesma escala do ground truth
    # Ground truth é contagem de eventos, predições são scores 0-100
    # Vamos trabalhar apenas com ranking
    
    # MAE na escala de eventos
    # Escalar predições para estimar contagem
    pred_scaled = predictions / 100.0 * gt_total.max()
    mae = np.mean(np.abs(pred_scaled - gt_total))
    
    # P@20
    top_20 = np.argsort(predictions)[-20:]
    p20 = (gt_total[top_20] > 0).sum() / 20
    
    # Coverage
    events_in_top20 = gt_total[top_20].sum()
    coverage = events_in_top20 / gt_total.sum() if gt_total.sum() > 0 else 0
    
    # Top-10
    top_10 = np.argsort(predictions)[-10:]
    p10 = (gt_total[top_10] > 0).sum() / 10
    
    # Nodes predicted
    threshold = np.percentile(predictions, 90)
    nodes_pred = (predictions > threshold).sum()
    
    print(f"\n{'='*60}")
    print(f"{method_name}")
    print(f"{'='*60}")
    print(f"P@10:         {p10:.2%}")
    print(f"P@20:         {p20:.2%}")
    print(f"Coverage:     {coverage:.2%} ({events_in_top20:.0f}/{gt_total.sum():.0f} eventos)")
    print(f"MAE (scaled): {mae:.4f}")
    print(f"Nós previstos (P90): {nodes_pred}")
    
    # Top-10 detalhado
    print(f"\nTop-10 Nós:")
    for rank, idx in enumerate(top_10[::-1], 1):
        real = gt_total[idx]
        pred = predictions[idx]
        hit = "✓" if real > 0 else " "
        print(f"  {rank}. Nó {idx}: score={pred:.2f}, real={real:.0f} {hit}")
    
    return {
        'method': method_name,
        'p10': p10,
        'p20': p20,
        'coverage': coverage,
        'mae': mae,
        'nodes_pred': nodes_pred
    }

def main():
    print(f"\n{'#'*80}")
    print("VALIDAÇÃO DO SISTEMA HÍBRIDO REAL (ST-GCN + RankingInference)")
    print(f"{'#'*80}")
    
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
    
    print(f"\nPeríodo: {all_dates[0]} a {all_dates[-1]}")
    print(f"Total eventos: {Y_true.sum():.0f}")
    print(f"Nós afetados: {(Y_true.sum(axis=1) > 0).sum()}")
    
    # Executar validação
    device = torch.device('cpu')
    predictions_dict = validate_hybrid_system(graph_data, Y_true, all_dates, device)
    
    # Avaliar cada variante
    print(f"\n{'='*80}")
    print("COMPARAÇÃO DE VARIANTES")
    print(f"{'='*80}")
    
    results = []
    
    # 1. ST-GCN puro (valores brutos)
    results.append(evaluate_predictions(
        predictions_dict['stgcn_only'], 
        Y_true, 
        "1. ST-GCN Puro (valores brutos)"
    ))
    
    # 2. ST-GCN com percentil
    results.append(evaluate_predictions(
        predictions_dict['stgcn_percentile'], 
        Y_true, 
        "2. ST-GCN + Percentil"
    ))
    
    # 3. Sistema Híbrido Final (ST-GCN + Ranking)
    results.append(evaluate_predictions(
        predictions_dict['hybrid_final'], 
        Y_true, 
        "3. HÍBRIDO FINAL (ST-GCN + Ranking 70/30)"
    ))
    
    # 4. Baseline MA3 para comparação
    cvli_3d = graph_data['node_features'][:, -3:, 0]
    baseline_pred = cvli_3d.mean(axis=1)
    # Normalizar baseline para 0-100 também
    baseline_norm = (baseline_pred / baseline_pred.max() * 100) if baseline_pred.max() > 0 else baseline_pred
    results.append(evaluate_predictions(
        baseline_norm, 
        Y_true, 
        "4. Baseline MA3 (comparação)"
    ))
    
    # Resumo final
    print(f"\n{'='*80}")
    print("RESUMO COMPARATIVO")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    print(f"\n{df_results[['method', 'p20', 'p10', 'coverage']].to_string(index=False)}")
    
    best_p20 = df_results.loc[df_results['p20'].idxmax()]
    
    print(f"\n🏆 MELHOR P@20: {best_p20['method']}")
    print(f"   P@20: {best_p20['p20']:.2%}")
    print(f"   P@10: {best_p20['p10']:.2%}")
    print(f"   Coverage: {best_p20['coverage']:.2%}")
    
    # Salvar
    output_file = os.path.join(BASE_DIR, 'reports', f'hybrid_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'period': {'start': str(all_dates[0]), 'end': str(all_dates[-1])},
            'results': [{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in r.items()} for r in results],
            'best_method': best_p20['method'],
            'conclusion': 'Validação com sistema híbrido real em produção'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Resultados salvos: {output_file}\n")

if __name__ == '__main__':
    main()

"""
Diagnóstico completo do modelo ST-GAT em dados recentes
Identifica problemas de treino e calibração
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

def analyze_model_output_distribution(predictions, ground_truth):
    """Analisa distribuição das predições vs realidade"""
    print(f"\n{'='*80}")
    print("ANÁLISE DE DISTRIBUIÇÃO")
    print(f"{'='*80}")
    
    print(f"\nPREDIÇÕES:")
    print(f"  Min:        {predictions.min():.4f}")
    print(f"  Max:        {predictions.max():.4f}")
    print(f"  Média:      {predictions.mean():.4f}")
    print(f"  Mediana:    {np.median(predictions):.4f}")
    print(f"  Std Dev:    {predictions.std():.4f}")
    print(f"  P5:         {np.percentile(predictions, 5):.4f}")
    print(f"  P25:        {np.percentile(predictions, 25):.4f}")
    print(f"  P75:        {np.percentile(predictions, 75):.4f}")
    print(f"  P95:        {np.percentile(predictions, 95):.4f}")
    
    print(f"\nGROUND TRUTH:")
    gt_total = ground_truth.sum(axis=1)
    print(f"  Min:        {gt_total.min():.4f}")
    print(f"  Max:        {gt_total.max():.4f}")
    print(f"  Média:      {gt_total.mean():.4f}")
    print(f"  Mediana:    {np.median(gt_total):.4f}")
    print(f"  Std Dev:    {gt_total.std():.4f}")
    print(f"  Nós com eventos: {(gt_total > 0).sum()}/{len(gt_total)}")
    
    # Problema de calibração
    print(f"\n⚠️ PROBLEMA IDENTIFICADO:")
    print(f"  Modelo prevê eventos em {(predictions > 0.5).sum()}/{len(predictions)} nós")
    print(f"  Realidade: eventos em {(gt_total > 0).sum()}/{len(gt_total)} nós")
    print(f"  Ratio: {(predictions > 0.5).sum() / max(1, (gt_total > 0).sum()):.1f}x mais nós previstos")

def test_different_thresholds(predictions, ground_truth):
    """Testa diferentes thresholds para otimizar métricas"""
    print(f"\n{'='*80}")
    print("OTIMIZAÇÃO DE THRESHOLD")
    print(f"{'='*80}")
    
    gt_total = ground_truth.sum(axis=1)
    
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    for th in thresholds:
        # Aplicar threshold
        pred_binary = (predictions > th).astype(int)
        
        # Precisão
        tp = ((pred_binary == 1) & (gt_total > 0)).sum()
        fp = ((pred_binary == 1) & (gt_total == 0)).sum()
        fn = ((pred_binary == 0) & (gt_total > 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # P@20
        top_20 = np.argsort(predictions)[-20:]
        p20 = (gt_total[top_20] > 0).sum() / 20
        
        results.append({
            'threshold': th,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'p@20': p20,
            'nodes_predicted': pred_binary.sum()
        })
    
    df_results = pd.DataFrame(results)
    print(f"\n{df_results.to_string(index=False)}")
    
    # Melhor threshold
    best_idx = df_results['f1'].idxmax()
    print(f"\n✓ MELHOR THRESHOLD: {df_results.loc[best_idx, 'threshold']}")
    print(f"  F1-Score: {df_results.loc[best_idx, 'f1']:.4f}")
    print(f"  Precision: {df_results.loc[best_idx, 'precision']:.4f}")
    print(f"  Recall: {df_results.loc[best_idx, 'recall']:.4f}")

def test_percentile_ranking(predictions, ground_truth):
    """Testa ranking por percentil ao invés de valores absolutos"""
    print(f"\n{'='*80}")
    print("RANKING POR PERCENTIL")
    print(f"{'='*80}")
    
    gt_total = ground_truth.sum(axis=1)
    
    # Top-k baseado em percentil
    for top_k in [5, 10, 20, 30]:
        top_indices = np.argsort(predictions)[-top_k:]
        hits = (gt_total[top_indices] > 0).sum()
        precision = hits / top_k
        
        # Cobertura dos eventos reais
        total_events = gt_total.sum()
        events_captured = gt_total[top_indices].sum()
        coverage = events_captured / total_events if total_events > 0 else 0
        
        print(f"\nTop-{top_k}:")
        print(f"  Precisão: {precision:.2%} ({hits}/{top_k} nós)")
        print(f"  Cobertura: {coverage:.2%} ({events_captured:.0f}/{total_events:.0f} eventos)")

def analyze_prediction_errors(predictions, ground_truth, nodes_gdf):
    """Analisa erros espacialmente"""
    print(f"\n{'='*80}")
    print("ANÁLISE DE ERROS")
    print(f"{'='*80}")
    
    gt_total = ground_truth.sum(axis=1)
    errors = predictions - gt_total
    
    # Tipos de erro
    false_positives = ((predictions > 0.5) & (gt_total == 0)).sum()
    false_negatives = ((predictions <= 0.5) & (gt_total > 0)).sum()
    true_positives = ((predictions > 0.5) & (gt_total > 0)).sum()
    true_negatives = ((predictions <= 0.5) & (gt_total == 0)).sum()
    
    print(f"\nMATRIZ DE CONFUSÃO (threshold=0.5):")
    print(f"  True Positives:  {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  True Negatives:  {true_negatives}")
    print(f"  False Negatives: {false_negatives}")
    
    # Maiores erros
    print(f"\nMAIORES SUPERESTIMAÇÕES:")
    top_overestimates = np.argsort(errors)[-5:][::-1]
    for idx in top_overestimates:
        print(f"  Nó {idx}: previsto={predictions[idx]:.2f}, real={gt_total[idx]:.0f}, erro=+{errors[idx]:.2f}")
    
    print(f"\nMAIORES SUBESTIMAÇÕES:")
    top_underestimates = np.argsort(errors)[:5]
    for idx in top_underestimates:
        print(f"  Nó {idx}: previsto={predictions[idx]:.2f}, real={gt_total[idx]:.0f}, erro={errors[idx]:.2f}")

def check_model_weights(model):
    """Verifica se pesos do modelo estão saudáveis"""
    print(f"\n{'='*80}")
    print("ANÁLISE DOS PESOS DO MODELO")
    print(f"{'='*80}")
    
    total_params = 0
    nan_params = 0
    inf_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        nan_params += torch.isnan(param).sum().item()
        inf_params += torch.isinf(param).sum().item()
        
        # Estatísticas por camada
        if 'weight' in name:
            print(f"\n{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
    
    print(f"\n{'='*60}")
    print(f"Total params: {total_params}")
    print(f"NaN params: {nan_params}")
    print(f"Inf params: {inf_params}")
    
    if nan_params > 0 or inf_params > 0:
        print(f"\n❌ PROBLEMA: Modelo tem pesos inválidos!")
        return False
    else:
        print(f"\n✓ Pesos do modelo estão saudáveis")
        return True

def compare_training_vs_validation(model, graph_data, device):
    """Compara distribuição de treino vs validação"""
    print(f"\n{'='*80}")
    print("TREINO vs VALIDAÇÃO")
    print(f"{'='*80}")
    
    # Pegar dados de treino (últimos 7 dias antes da validação)
    train_window = graph_data['node_features'][:, -19:-12, :]
    val_window = graph_data['node_features'][:, -12:, :]
    
    # Fazer predição em ambos
    model.eval()
    
    def predict_window(window_data):
        adj_geo = graph_data['adj_geo']
        adj_conflict = graph_data['adj_conflict']
        
        def normalize_adj(adj_matrix):
            adj_tensor = torch.FloatTensor(adj_matrix)
            deg = adj_tensor.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
            norm_adj = deg_inv_sqrt.unsqueeze(1) * adj_tensor * deg_inv_sqrt.unsqueeze(0)
            return norm_adj
        
        norm_adj_geo = normalize_adj(adj_geo)
        norm_adj_conflict = normalize_adj(adj_conflict)
        adj_list = [norm_adj_geo.to(device), norm_adj_conflict.to(device)]
        
        X = torch.FloatTensor(np.transpose(window_data, (2, 0, 1))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(X, adj_list)
        
        return pred.squeeze().cpu().numpy()
    
    train_pred = predict_window(train_window)
    val_pred = predict_window(val_window)
    
    print(f"\nPREDIÇÕES EM DADOS DE TREINO (7 dias antes):")
    print(f"  Média: {train_pred.mean():.4f}")
    print(f"  Total: {train_pred.sum():.2f}")
    
    print(f"\nPREDIÇÕES EM DADOS DE VALIDAÇÃO (últimos 12 dias):")
    print(f"  Média: {val_pred.mean():.4f}")
    print(f"  Total: {val_pred.sum():.2f}")
    
    # Dados históricos reais
    train_real = train_window[:, :, 0].sum()
    val_real = val_window[:, :, 0].sum()
    
    print(f"\nDADOS REAIS:")
    print(f"  Treino: {train_real:.0f} eventos")
    print(f"  Validação: {val_real:.0f} eventos")
    
    if abs(train_pred.sum() - val_pred.sum()) / max(train_pred.sum(), 1) > 0.5:
        print(f"\n⚠️ POSSÍVEL OVERFITTING: Grande diferença entre treino e validação")

def main():
    print(f"\n{'#'*80}")
    print("DIAGNÓSTICO DE DESEMPENHO DO MODELO ST-GAT")
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
    
    # Carregar modelo
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
    
    # Fazer predição
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
        predictions = model(X, adj_list).squeeze().cpu().numpy()
    
    print(f"\n{'='*80}")
    print("RESUMO DOS DADOS")
    print(f"{'='*80}")
    print(f"Período: {all_dates[0]} a {all_dates[-1]}")
    print(f"Total eventos reais: {Y_true.sum():.0f}")
    print(f"Total previsto: {predictions.sum():.2f}")
    print(f"Ratio: {predictions.sum() / Y_true.sum():.2f}x")
    
    # Executar diagnósticos
    analyze_model_output_distribution(predictions, Y_true)
    check_model_weights(model)
    test_percentile_ranking(predictions, Y_true)
    test_different_thresholds(predictions, Y_true)
    analyze_prediction_errors(predictions, Y_true, nodes_gdf)
    compare_training_vs_validation(model, graph_data, device)
    
    # RESUMO FINAL
    print(f"\n{'='*80}")
    print("CONCLUSÕES E RECOMENDAÇÕES")
    print(f"{'='*80}")
    
    gt_total = Y_true.sum(axis=1)
    top_20_pred = np.argsort(predictions)[-20:]
    p20 = (gt_total[top_20_pred] > 0).sum() / 20
    
    print(f"\n📊 MÉTRICAS PRINCIPAIS:")
    print(f"  P@20: {p20:.2%}")
    print(f"  Superestimação: {predictions.sum() / Y_true.sum():.1f}x")
    
    if predictions.sum() / Y_true.sum() > 10:
        print(f"\n❌ PROBLEMA CRÍTICO: Superestimação severa (>{predictions.sum() / Y_true.sum():.0f}x)")
        print(f"\n💡 CAUSAS PROVÁVEIS:")
        print(f"  1. Loss function não penaliza suficientemente falsos positivos")
        print(f"  2. Modelo não aprendeu esparsidade dos eventos")
        print(f"  3. Dados de treino desbalanceados")
        print(f"  4. Falta de regularização")
        
        print(f"\n🔧 SOLUÇÕES SUGERIDAS:")
        print(f"  1. IMEDIATA: Usar threshold adaptativo (percentil 95+)")
        print(f"  2. CURTO PRAZO: Retreinar com Focal Loss (gamma=2.0)")
        print(f"  3. MÉDIO PRAZO: Aumentar dropout (0.3 → 0.5)")
        print(f"  4. LONGO PRAZO: Reformular como classificação + regressão")
    
    elif p20 < 0.3:
        print(f"\n⚠️ PROBLEMA: Baixa precisão no ranking")
        print(f"\n💡 SOLUÇÕES:")
        print(f"  1. Combinar com modelo de sazonalidade")
        print(f"  2. Adicionar features de tendência")
        print(f"  3. Usar ensemble com baseline MA3")
    
    # Salvar diagnóstico
    output_file = os.path.join(BASE_DIR, 'reports', f'model_diagnosis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    print(f"\n{'='*80}")
    print(f"✓ Diagnóstico completo salvo em: {output_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

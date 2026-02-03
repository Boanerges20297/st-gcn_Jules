#!/usr/bin/env python3
"""
Script de Avaliação do Modelo ST-GCN v3
Testa eficácia com 8 canais e janela de 30 dias
"""

import os
import sys
import pickle
import numpy as np
import torch
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import STGCN
from numpy.lib.stride_tricks import sliding_window_view

# Configuração
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model_v2.pth')
HISTORY_WINDOW = 30
BATCH_SIZE = 32

def normalize_adj(adj_np):
    adj_t = torch.FloatTensor(adj_np)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt)

def precision_at_k(pred, target, k=5):
    """Precision@K com tratamento de dias sem eventos"""
    batch_size = pred.shape[0]
    p_k_sum = 0.0
    valid_samples = 0

    for i in range(batch_size):
        p = pred[i, :, 0].detach().cpu().numpy()
        t = target[i, :, 0].detach().cpu().numpy()

        if t.max() == 0:
            # Skip dias sem eventos reais
            continue
        
        valid_samples += 1
        _, true_top_k_indices = torch.topk(torch.FloatTensor(t), min(k, len(t)))
        true_top_k_indices = true_top_k_indices.numpy()
        
        pred_top_k = torch.topk(torch.FloatTensor(p), min(k, len(p)))[1].numpy()
        
        hits = len(set(true_top_k_indices) & set(pred_top_k))
        p_k_sum += (hits / min(k, (t > 0).sum()))

    return p_k_sum / max(1, valid_samples)

def evaluate_model():
    print("=" * 80)
    print("AVALIAÇÃO DE EFICÁCIA - ST-GCN v3 (30 dias, 8 canais)")
    print("=" * 80)
    
    # Carregar dados
    print("\n[1/5] Carregando dados...")
    if not os.path.exists(DATA_FILE):
        print(f"❌ ERRO: {DATA_FILE} não encontrado!")
        return
    
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)
    
    node_features = data_pack['node_features']
    adj_geo = data_pack['adj_geo']
    adj_faction = data_pack['adj_conflict']
    dates = data_pack['dates']
    feature_names = data_pack.get('feature_names', ['CVLI', 'CVP', 'Tension', 'DOW_sin', 'DOW_cos', 'MONTH_sin', 'MONTH_cos', 'IS_WEEKEND'])
    
    print(f"✅ Dados carregados:")
    print(f"   - Shape: {node_features.shape} (nós, dias, features)")
    print(f"   - Período: {dates[0].date()} até {dates[-1].date()}")
    print(f"   - Features: {feature_names}")
    
    # Preparar datasets
    print(f"\n[2/5] Preparando datasets com janela={HISTORY_WINDOW}...")
    windows = sliding_window_view(node_features, HISTORY_WINDOW, axis=1)
    X = windows[:, :-1, :, :]  # (Nodes, Samples, Features, WindowSize)
    Y = node_features[:, HISTORY_WINDOW:, 0:1]  # (Nodes, Samples, 1)
    
    X = X.transpose(1, 2, 0, 3)  # (Samples, Features, Nodes, WindowSize)
    Y = Y.transpose(1, 0, 2)  # (Samples, Nodes, 1)
    
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]
    
    print(f"✅ Dataset preparado:")
    print(f"   - Treino: {X[:split_idx].shape}")
    print(f"   - Teste: {X_test.shape}")
    
    # Carregar modelo
    print(f"\n[3/5] Carregando modelo...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: {MODEL_PATH} não encontrado!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]
    
    print(f"   - Modelo: STGCN(nodes={num_nodes}, in_channels={num_features}, window={HISTORY_WINDOW})")
    print(f"   - Device: {device}")
    
    norm_adj_geo = normalize_adj(adj_geo)
    norm_adj_faction = normalize_adj(adj_faction)
    norm_adj_list = [norm_adj_geo, norm_adj_faction]
    norm_adj_list = [a.to(device) for a in norm_adj_list]
    
    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=HISTORY_WINDOW, num_classes=1, num_graphs=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Modelo carregado de {MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERRO ao carregar modelo: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Avaliar
    print(f"\n[4/5] Avaliando no conjunto de teste ({len(X_test)} amostras)...")
    
    all_preds = []
    all_targets = []
    all_p5 = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(X_test))
            batch_x = torch.FloatTensor(X_test[i:batch_end]).to(device)
            batch_y = torch.FloatTensor(Y_test[i:batch_end]).to(device)
            
            output = model(batch_x, norm_adj_list)
            
            all_preds.append(output.detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())
            
            p5 = precision_at_k(output, batch_y, k=5)
            all_p5.append(p5)
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"✅ Predições concluídas")
    
    # Calcular métricas
    print(f"\n[5/5] Calculando métricas...")
    
    # P@5
    mean_p5 = np.mean(all_p5)
    
    # MAE
    mae = np.mean(np.abs(preds - targets))
    
    # RMSE
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # Correlação entre predições e reais
    pred_flat = preds.flatten()
    target_flat = targets.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Acurácia de "zero" (modelo prevê zero quando deveria)
    zero_accuracy = np.mean((pred_flat < 0.5) == (target_flat < 0.5))
    
    # Top-10 accuracy
    top10_accuracy = 0
    for day_idx in range(len(Y_test)):
        true_vals = targets[day_idx, :, 0]
        pred_vals = preds[day_idx, :, 0]
        
        if true_vals.max() == 0:
            continue
        
        true_top10 = set(np.argsort(true_vals)[-10:])
        pred_top10 = set(np.argsort(pred_vals)[-10:])
        hits = len(true_top10 & pred_top10)
        top10_accuracy += hits / 10
    
    top10_accuracy /= len(Y_test)
    
    # Estatísticas descritivas
    print("\n" + "=" * 80)
    print("RESULTADOS")
    print("=" * 80)
    print(f"\n📊 MÉTRICAS DE ACURÁCIA:")
    print(f"   Precision@5 (P@5):        {mean_p5:.4f} ({mean_p5*100:.2f}%)")
    print(f"   Top-10 Accuracy:          {top10_accuracy:.4f} ({top10_accuracy*100:.2f}%)")
    print(f"   Zero-Detection Accuracy:  {zero_accuracy:.4f} ({zero_accuracy*100:.2f}%)")
    
    print(f"\n📉 ERROS:")
    print(f"   MAE (Mean Absolute Error):  {mae:.6f}")
    print(f"   RMSE (Root Mean Squared):   {rmse:.6f}")
    print(f"   Correlação:                 {correlation:.4f}")
    
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   Predições - Min: {pred_flat.min():.4f}, Max: {pred_flat.max():.4f}, Mean: {pred_flat.mean():.4f}")
    print(f"   Reais      - Min: {target_flat.min():.4f}, Max: {target_flat.max():.4f}, Mean: {target_flat.mean():.4f}")
    
    # Analise por feature
    print(f"\n🔍 ANÁLISE POR FEATURE:")
    for ch in range(min(3, num_features)):
        ch_data = node_features[:, -HISTORY_WINDOW:, ch]
        print(f"   Canal {ch} ({feature_names[ch]}): min={ch_data.min():.4f}, max={ch_data.max():.4f}, mean={ch_data.mean():.6f}")
    
    # Conclusões
    print(f"\n💡 CONCLUSÕES:")
    if mean_p5 >= 0.15:
        print(f"   ✅ P@5 = {mean_p5*100:.2f}% - EXCELENTE! Modelo é viável para produção")
    elif mean_p5 >= 0.12:
        print(f"   ⚠️  P@5 = {mean_p5*100:.2f}% - BOM! Modelo é aceitável com monitoramento")
    elif mean_p5 >= 0.10:
        print(f"   ⚠️  P@5 = {mean_p5*100:.2f}% - ACEITÁVEL! Melhorias necessárias")
    else:
        print(f"   ❌ P@5 = {mean_p5*100:.2f}% - INSUFICIENTE! Revisar features/arquitetura")
    
    print(f"\n   - Zero detection: {'✅ Bom' if zero_accuracy > 0.75 else '⚠️  Precisa melhorar'}")
    print(f"   - Correlação:     {'✅ Forte' if correlation > 0.5 else '⚠️  Fraca'}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    evaluate_model()

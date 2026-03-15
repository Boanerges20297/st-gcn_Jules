import torch
import numpy as np
import os
import pandas as pd
from src.core.orchestrator import StateOrchestrator, normalize_name

def get_in_channels(model_path):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    return sd['layer1.time_conv.weight'].shape[1]

def test_specific_model(orch, model_path):
    in_channels = get_in_channels(model_path)
    print(f"\n--- Testando Modelo: {os.path.basename(model_path)} ({in_channels} Canais) ---")
    
    spec = orch.specialists['fortaleza']
    data = spec['data']
    num_nodes = len(data['nodes_gdf'])
    window = 120 if in_channels >= 32 else 90
    
    # Carregar o modelo específico no orquestrador temporariamente
    model = spec['model'].__class__(num_nodes=num_nodes, in_channels=in_channels, time_steps=window).to(orch.device)
    ckpt = torch.load(model_path, map_location=orch.device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # --- Cálculo de Risco (Lógica simplificada do Orchestrator) ---
    extra_history = 60 if in_channels >= 32 else 0
    total_window = window + extra_history
    x_raw_extended = data['node_features'][:, -total_window:, :].copy()
    
    if in_channels >= 32:
        momentum_feat = np.zeros((num_nodes, total_window, in_channels - 29))
        cold_streak = np.zeros(num_nodes)
        for t in range(60, total_window):
            # 7d
            momentum_feat[:, t, 0] = x_raw_extended[:, t-7:t, 0].sum(axis=1) - x_raw_extended[:, t-14:t-7, 0].sum(axis=1)
            # 14d
            momentum_feat[:, t, 1] = x_raw_extended[:, t-14:t, 0].sum(axis=1) - x_raw_extended[:, t-28:t-14, 0].sum(axis=1)
            # 30d
            momentum_feat[:, t, 2] = x_raw_extended[:, t-30:t, 0].sum(axis=1) - x_raw_extended[:, t-60:t-30, 0].sum(axis=1)
            if in_channels == 33:
                crimes_today = x_raw_extended[:, t, 0]
                cold_streak = np.where(crimes_today > 0, 0, cold_streak + 1)
                momentum_feat[:, t, 3] = np.clip(cold_streak, 0, 30)
        
        x_raw_extended = np.concatenate([x_raw_extended, momentum_feat], axis=2)
        # REMOVIDO: Normalização Z-Score (Os modelos de Elite esperam Sinais Brutos para preservar picos)
        # for c_idx in range(29, in_channels):
        #     m_mean = x_raw_extended[:, :, c_idx].mean()
        #     m_std = x_raw_extended[:, :, c_idx].std() + 1e-6
        #     x_raw_extended[:, :, c_idx] = (x_raw_extended[:, :, c_idx] - m_mean) / m_std

    x_final = x_raw_extended[:, -window:, :].copy()
    x_tensor = torch.from_numpy(x_final).float().permute(2, 0, 1).unsqueeze(0).to(orch.device)
    adj = orch._norm_adj(data['adj_geo'], data['adj_conflict'])
    
    with torch.no_grad():
        out = model(x_tensor, adj).squeeze().cpu().numpy()
    
    # Ranking
    final_logits = out + (data['nodes_gdf']['tension_index'].values.astype(float) * 0.5)
    s = 1 / (1 + np.exp(-0.7 * (final_logits - (-1.0))))
    scores = np.clip(s * 100, 5.0, 100.0)
    
    name_to_idx = {normalize_name(row['name']): i for i, row in data['nodes_gdf'].iterrows()}
    sorted_indices = np.argsort(scores)[::-1]
    top_10_indices = sorted_indices[:10]
    
    # Verificação nos últimos 28 dias para maior estabilidade estatística
    history_p10 = []
    node_features = data['node_features']
    print(f"  🔍 Avaliando últimos 28 dias...")
    for d in range(-1, -29, -1):
        actual_crimes = node_features[:, d, 0]
        hits = np.sum(actual_crimes[top_10_indices] > 0)
        p_at_10 = (hits / 10) * 100
        history_p10.append(p_at_10)
    
    avg_p10 = np.mean(history_p10)
    print(f"  >>> Média Final P@10: {avg_p10:.2f}%")
    return avg_p10

def main():
    orch = StateOrchestrator(os.getcwd())
    models = [
        'models/active/fortaleza_model_active.pth',
        'models/active/fortaleza_retrain_64.pth'
    ]
    
    for m in models:
        if os.path.exists(m):
            test_specific_model(orch, m)
        else:
            print(f"❌ Arquivo não encontrado: {m}")

if __name__ == "__main__":
    main()

import torch
import numpy as np
import os
from src.core.orchestrator import StateOrchestrator

def inspect_tensor_values():
    orch = StateOrchestrator(os.getcwd())
    spec = orch.specialists['fortaleza']
    data = spec['data']
    window = spec['window']
    channels = spec['channels']
    
    extra_history = 60 if channels >= 32 else 0
    total_window = window + extra_history
    x_raw = data['node_features'][:, -total_window:, :].copy()
    
    print(f"--- Inspeção de Tensores (Fortaleza - {channels} Canais) ---")
    print(f"Shape inicial: {x_raw.shape}")
    
    # Simular o cálculo do Momentum exatamente como no Orchestrator
    num_nodes = x_raw.shape[0]
    momentum_feat = np.zeros((num_nodes, total_window, channels - 29))
    cold_streak = np.zeros(num_nodes)
    
    for t in range(60, total_window):
        # Escala 1 (7 dias)
        recent_7 = x_raw[:, t-7:t, 0].sum(axis=1)
        past_7 = x_raw[:, t-14:t-7, 0].sum(axis=1)
        momentum_feat[:, t, 0] = recent_7 - past_7
        
        # Escala 2 (14 dias)
        recent_14 = x_raw[:, t-14:t, 0].sum(axis=1)
        past_14 = x_raw[:, t-28:t-14, 0].sum(axis=1)
        momentum_feat[:, t, 1] = recent_14 - past_14
        
        # Escala 3 (30 dias)
        recent_30 = x_raw[:, t-30:t, 0].sum(axis=1)
        past_30 = x_raw[:, t-60:t-30, 0].sum(axis=1)
        momentum_feat[:, t, 2] = recent_30 - past_30
        
        if channels == 33:
            crimes_today = x_raw[:, t, 0]
            cold_streak = np.where(crimes_today > 0, 0, cold_streak + 1)
            momentum_feat[:, t, 3] = np.clip(cold_streak, 0, 30)

    print(f"Momentum (Média/Max) Canal 29 (7d):  {momentum_feat[:, :, 0].mean():.4f} / {momentum_feat[:, :, 0].max():.4f}")
    print(f"Momentum (Média/Max) Canal 30 (14d): {momentum_feat[:, :, 1].mean():.4f} / {momentum_feat[:, :, 1].max():.4f}")
    print(f"Momentum (Média/Max) Canal 31 (30d): {momentum_feat[:, :, 2].mean():.4f} / {momentum_feat[:, :, 2].max():.4f}")
    if channels == 33:
        print(f"Momentum (Média/Max) Canal 32 (Cold): {momentum_feat[:, :, 3].mean():.4f} / {momentum_feat[:, :, 3].max():.4f}")

    # Checar se houve normalização agressiva
    x_combined = np.concatenate([x_raw, momentum_feat], axis=2)
    for c in range(29, channels):
        mean = x_combined[:, :, c].mean()
        std = x_combined[:, :, c].std() + 1e-6
        print(f"Estatísticas Canal {c} -> Mean: {mean:.4f}, Std: {std:.4f}")

    # Verificar se o modelo está dando scores muito baixos ou uniformes
    results = orch.get_combined_risk()
    scores = list(results.values())
    print(f"\nScores de Predição: Min={min(scores):.2f}, Max={max(scores):.2f}, Mean={np.mean(scores):.2f}")
    
    # Verificar os últimos crimes reais
    last_crimes = x_raw[:, -7:, 0].sum(axis=1)
    nodes_with_crimes = np.where(last_crimes > 0)[0]
    print(f"Nós que tiveram crimes nos últimos 7 dias: {nodes_with_crimes}")
    for idx in nodes_with_crimes:
        name = data['nodes_gdf'].iloc[idx]['name']
        score = results.get(orch.normalize_name(name), 0)
        print(f"  - {name}: Score={score:.2f}")

if __name__ == "__main__":
    inspect_tensor_values()

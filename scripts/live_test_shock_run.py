import torch
import pickle
import numpy as np
import os
import sys
import pandas as pd

# Adiciona Phase4 ao path para importar o modelo
sys.path.append(os.path.join(os.getcwd(), 'Phase4'))
from model_v4 import DeepSTGAT

def live_test():
    print("--- INICIANDO LIVE TEST (CHECKPOINT SHOCK RUN) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('data/processed/processed_graph_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    adj_list = [torch.from_numpy(data['adj_geo']).float().to(device), 
                torch.from_numpy(data['adj_conflict']).float().to(device)]
    
    node_features = data['node_features']
    num_nodes = node_features.shape[0]
    in_channels = 27
    history_window = 30 
    
    model = DeepSTGAT(num_nodes=num_nodes, in_channels=in_channels, time_steps=history_window).to(device)
    
    model_path = 'models/phase5/best_stgat_v5_massive.pth'
    if not os.path.exists(model_path):
        print("Erro: Modelo nao encontrado")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo carregado (P@10 Val: {checkpoint.get('p10', 0):.4f})")

    last_window = node_features[:, -history_window-7:-7, :]
    x = torch.from_numpy(last_window).float().permute(2, 0, 1).unsqueeze(0).to(device)
    y_real = node_features[:, -7:, 0].sum(axis=1) 
    
    with torch.no_grad():
        out = model(x, adj_list).squeeze().cpu().numpy()

    top_10_idx = np.argsort(out)[-10:][::-1]
    
    print("\n--- TOP 10 BAIRROS DE MAIOR RISCO (PREDITO) ---")
    print(f"{'Ranking':<5} | {'Bairro':<25} | {'Score':<8} | {'Real CVLI (7d)':<12}")
    print("-" * 65)
    
    hits = 0
    for i, idx in enumerate(top_10_idx):
        name = data['nodes_gdf'].iloc[idx]['name']
        score = out[idx]
        real = y_real[idx]
        status = "✅ ACERTO" if real > 0 else ""
        if real > 0: hits += 1
        print(f"#{i+1:<4} | {name:<25} | {score:8.4f} | {real:<12.0f} {status}")

    real_positive_indices = np.where(y_real > 0)[0]
    missed = [idx for idx in real_positive_indices if idx not in top_10_idx]
    
    print(f"\nResumo: {hits} acertos no Top 10.")
    print(f"Total de bairros com CVLI no periodo: {len(real_positive_indices)}")
    
    if missed:
        print("\nBairros com CVLI que ficaram fora do Top 10 (Ranking do Modelo):")
        for idx in missed[:5]:
            name = data['nodes_gdf'].iloc[idx]['name']
            rank_pos = np.where(np.argsort(out)[::-1] == idx)[0][0] + 1
            print(f"  - {name:<25} (Posicao no Ranking: #{rank_pos})")

if __name__ == "__main__":
    live_test()

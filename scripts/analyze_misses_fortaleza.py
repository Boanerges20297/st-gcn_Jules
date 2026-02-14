import torch
import pickle
import numpy as np
import os
import sys
import pandas as pd

# Adiciona Phase4 ao path para importar o modelo
sys.path.append(os.path.join(os.getcwd(), 'Phase4'))
from model_v4 import DeepSTGAT

def analyze_misses():
    print("--- ANALISE DE ERRO: QUEM O MODELO PERDEU E POR QUE? ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('data/processed/processed_graph_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    def normalize_adj(adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    adj_list = [torch.from_numpy(normalize_adj(data['adj_geo'])).float().to(device), 
                torch.from_numpy(normalize_adj(data['adj_conflict'])).float().to(device)]
    
    node_features = data['node_features']
    
    model = DeepSTGAT(num_nodes=node_features.shape[0], in_channels=28, time_steps=30).to(device)
    checkpoint = torch.load('models/phase5/best_stgat_v5_massive.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    last_window = node_features[:, -37:-7, :]
    y_real = node_features[:, -7:, 0].sum(axis=1) 
    
    with torch.no_grad():
        out = model(torch.from_numpy(last_window).float().permute(2, 0, 1).unsqueeze(0).to(device), adj_list).squeeze().cpu().numpy()

    real_positives = np.where(y_real > 0)[0]
    top_10 = np.argsort(out)[-10:]
    missed = [idx for idx in real_positives if idx not in top_10]
    
    print("\nBairros que tiveram crime mas estao FORA do Top 10:")
    for idx in missed:
        name = data['nodes_gdf'].iloc[idx]['name']
        rank = np.where(np.argsort(out)[::-1] == idx)[0][0] + 1
        score = out[idx]
        
        print(f"\n>>> {name} (Posicao #{rank}, Score {score:.4f})")
        
        prev_cvli = node_features[idx, -37:-7, 0].sum()
        veiculos = node_features[idx, -37:-7, 1].sum()
        incursao = node_features[idx, -37:-7, 26].sum()
        momentum = node_features[idx, -37:-7, 24].mean()
        
        print(f"    - Hist. Recente CVLI: {prev_cvli}")
        print(f"    - Roubo de Veiculos: {veiculos}")
        print(f"    - Alertas de Incursao: {incursao}")
        print(f"    - Media Momentum 7d: {momentum:.4f}")
        
        if veiculos == 0 and prev_cvli == 0:
            print("    MOTIVO PROVAVEL: Bairro estava frio (sem sinais logisticos ou historicos).")
        elif rank > 50:
            print("    MOTIVO PROVAVEL: Modelo esta subestimando o peso da Incursao ou Momentum.")
        else:
            print("    MOTIVO PROVAVEL: Bairro esta subindo no ranking, mas ainda nao rompeu o Top 10.")

if __name__ == "__main__":
    analyze_misses()

import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def analyze():
    print("--- ANALISE DE CORRELACAO CVP-VEICULO VS CVLI (1001 DIAS) ---")
    
    with open('data/processed/processed_graph_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    feats = data['node_features']
    adj_geo = data['adj_geo']
    
    num_nodes = feats.shape[0]
    num_days = feats.shape[1]
    
    cvli_all = feats[:, :, 0]
    veic_all = feats[:, :, 25]
    
    # 1. Probabilidade Base
    prob_base = (cvli_all > 0).sum() / (num_nodes * num_days)
    print(f"Probabilidade Base de CVLI: {prob_base:.5f}")
    
    # 2. Probabilidade Condicional (Janela de 7 dias)
    hits = 0
    total_veic_events = 0
    
    for n in range(num_nodes):
        veic_days = np.where(veic_all[n] > 0)[0]
        for d in veic_days:
            if d + 7 < num_days:
                total_veic_events += 1
                if np.any(cvli_all[n, d+1 : d+8] > 0):
                    hits += 1
    
    if total_veic_events > 0:
        prob_cond = hits / total_veic_events
        print(f"Probabilidade de CVLI nos 7 dias APOS um Roubo de Veiculo no mesmo bairro: {prob_cond:.5f}")
        # Estimativa de aumento de risco (considerando que o alvo tem 7 chances de ocorrer)
        baseline_7d = 1 - (1 - prob_base)**7
        print(f"Aumento de Risco (Lift): {prob_cond / baseline_7d:.2f}x")
    
    # 3. Correlação de Pearson com Lags
    lags = [0, 1, 2, 3, 7]
    print("\nCorrelacao de Pearson por Lag (Veiculo -> CVLI):")
    
    for lag in lags:
        if lag == 0:
            corr, _ = pearsonr(veic_all.flatten(), cvli_all.flatten())
        else:
            v_shifted = veic_all[:, :-lag].flatten()
            c_target = cvli_all[:, lag:].flatten()
            corr, _ = pearsonr(v_shifted, c_target)
        print(f"  Lag {lag}d: {corr:.4f}")

    # 4. Analise Espacial (Vizinhos)
    hits_spatial = 0
    for n in range(num_nodes):
        neighbors = np.where(adj_geo[n] > 0)[0]
        veic_days = np.where(veic_all[n] > 0)[0]
        for d in veic_days:
            if d + 7 < num_days:
                if np.any(cvli_all[neighbors, d+1 : d+8] > 0):
                    hits_spatial += 1
    
    if total_veic_events > 0:
        prob_spatial = hits_spatial / total_veic_events
        baseline_spatial_7d = 1 - (1 - prob_base)**(7 * 10) # Assumindo avg 10 vizinhos
        print(f"\nProbabilidade de CVLI em Bairros VIZINHOS apos Roubo de Veiculo: {prob_spatial:.5f}")

if __name__ == "__main__":
    analyze()

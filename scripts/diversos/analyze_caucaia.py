import pickle
import numpy as np
import os

def analyze_caucaia():
    path = 'data/processed/processed_rmf.pkl'
    if not os.path.exists(path):
        print("Arquivo nao encontrado.")
        return
        
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    nodes_gdf = data['nodes_gdf']
    node_features = data['node_features']
    
    idx = [i for i, n in enumerate(nodes_gdf['name']) if 'CAUCAIA' in n][0]
    row = nodes_gdf.iloc[idx]
    feats = node_features[idx, :, :]
    
    print("--- DIAGNOSTICO TATICO: " + str(row['name']) + " ---")
    print("Faccao: " + str(row.get('faction', 'N/A')))
    print("Indice de Tensao Geografica: " + str(row.get('tension_index', 0)))
    
    # Canais de Crime
    print("\n[HISTORICO 1000 DIAS]")
    print("Total CVLI: " + str(feats[:, 0].sum()))
    print("Total Roubo Veiculos: " + str(feats[:, 1].sum()))
    
    print("\n[JANELA RECENTE (30 DIAS)]")
    recent_cvli = feats[-30:, 0].sum()
    recent_veh = feats[-30:, 1].sum()
    print("CVLI Recente: " + str(recent_cvli))
    print("Veiculos Recente: " + str(recent_veh))
    
    # Canais de Inteligencia (23, 24, 25)
    print("\n[CHOQUES DE INTELIGENCIA ATIVOS]")
    print("Canal 23 (Supressao): " + str(feats[-1, 23]))
    print("Canal 24 (Tensao Normal): " + str(feats[-1, 24]))
    print("Canal 25 (Critico/Exogeno): " + str(feats[-1, 25]))
    
    # Analise de Momentum
    print("\n[MOMENTUM]")
    avg_7d = feats[-7:, 0].mean()
    avg_prev_7d = feats[-14:-7, 0].mean()
    print("Media 7 dias atual: " + str(round(float(avg_7d), 4)))
    print("Media 7 dias anterior: " + str(round(float(avg_prev_7d), 4)))
    
    if recent_cvli == 0:
        print("\nAVISO: O risco esta baixo porque NAO HA registros de crimes fatais (CVLI) vinculados a Caucaia nos ultimos 30 dias da base.")

if __name__ == "__main__":
    analyze_caucaia()

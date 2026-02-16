import pickle
import pandas as pd
import numpy as np

def fix_caucaia_metadata():
    path = 'data/processed/processed_rmf.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    nodes_gdf = data['nodes_gdf']
    
    # Encontrar o índice da Caucaia
    idx = nodes_gdf[nodes_gdf['name'].str.contains('CAUCAIA', na=False)].index[0]
    
    print(f"Antigo: Facção={nodes_gdf.at[idx, 'faction']}, Tensão={nodes_gdf.at[idx, 'tension_index']}")
    
    # Atualizar para refletir a disputa CV vs MASSA (17 polígonos ativos)
    nodes_gdf.at[idx, 'faction'] = 'CV'
    nodes_gdf.at[idx, 'tension_index'] = 1.0 # Máxima tensão por disputa territorial
    
    print(f"Novo: Facção={nodes_gdf.at[idx, 'faction']}, Tensão={nodes_gdf.at[idx, 'tension_index']}")
    
    # Salvar de volta
    data['nodes_gdf'] = nodes_gdf
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ Metadados de Caucaia atualizados com sucesso no RMF!")

if __name__ == "__main__":
    fix_caucaia_metadata()

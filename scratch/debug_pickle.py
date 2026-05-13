import pickle
import pandas as pd
import sys

path = r'c:\Users\Boanerges\Desktop\Projetos\Report Preview\data\processed\processed_interior.pkl'

try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("Sucesso ao carregar com pickle.load")
    print("Chaves:", data.keys())
    if 'nodes_gdf' in data:
        gdf = data['nodes_gdf']
        print("GDF tipo:", type(gdf))
        print("GDF head:", gdf.head())
except Exception as e:
    print(f"Erro no diagnóstico: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

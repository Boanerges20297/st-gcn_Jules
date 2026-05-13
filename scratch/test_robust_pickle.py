import pickle
import pandas as pd
import sys
import os

path = r'c:\Users\Boanerges\Desktop\Projetos\Report Preview\data\processed\processed_interior.pkl'

class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'pandas' in module and 'StringDtype' in name:
            try:
                from pandas import StringDtype
                return StringDtype
            except ImportError:
                return object
        return super().find_class(module, name)

try:
    with open(path, 'rb') as f:
        data = RobustUnpickler(f).load()
    print("Sucesso ao carregar com RobustUnpickler!")
    print("Chaves:", data.keys())
    if 'nodes_gdf' in data:
        gdf = data['nodes_gdf']
        print("GDF tipo:", type(gdf))
        print("GDF shape:", gdf.shape)
        print("GDF columns:", gdf.columns.tolist())
except Exception as e:
    print(f"Erro no teste robusto: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

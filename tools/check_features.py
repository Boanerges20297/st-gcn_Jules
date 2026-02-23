import pickle
import os

path = 'data/processed/processed_fortaleza.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

# Tentar encontrar os nomes das features
if 'feature_names' in data:
    print("--- NOMES DAS FEATURES ---")
    for i, name in enumerate(data['feature_names']):
        print(f"[{i}] {name}")
else:
    print("ATENCAO: 'feature_names' nao encontrado no pickle. Verificando estrutura...")
    print(data.keys())

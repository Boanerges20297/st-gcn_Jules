import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

RAW_PATH = 'data/raw/dados_status_ocorrencias_gerais.json'
OUT_PATH = 'data/processed/processed_graph_data_dense.pkl'

# Carrega dados brutos
with open(RAW_PATH, encoding='utf-8') as f:
    raw = json.load(f)

# Filtra eventos CVLI
events = [d for d in raw if 'tipo' in d and d['tipo'] == 'cvli']

# Agrupa por data
dates = sorted(set(d['data'] for d in events))

# Seleciona datas densas (com pelo menos 10 eventos)
dense_dates = [date for date in dates if sum(1 for d in events if d['data'] == date) >= 10]

# Cria tensor node_features (mock: apenas CVLI count por bairro)
bairros = sorted(set(d['bairro'] for d in events if d['bairro']))
if not bairros:
    bairros = sorted(set(d['cidade'] for d in events if d['cidade']))

num_nodes = len(bairros)
num_dates = len(dense_dates)
num_features = 26  # Mantém compatível

bairro_idx = {b: i for i, b in enumerate(bairros)}

tensor = np.zeros((num_nodes, num_dates, num_features), dtype=np.float32)

for t, date in enumerate(dense_dates):
    for d in events:
        if d['data'] == date:
            idx = bairro_idx.get(d['bairro']) or bairro_idx.get(d['cidade'])
            if idx is not None:
                tensor[idx, t, 0] += 1  # CVLI count

# Mock adjacências
adj_geo = np.eye(num_nodes)
adj_conflict = np.eye(num_nodes)

# Salva
data_pack = {
    'node_features': tensor,
    'adj_geo': adj_geo,
    'adj_conflict': adj_conflict,
    'dates': dense_dates,
    'feature_names': ['cvli'] + [f'f{i}' for i in range(1, num_features)],
    'nodes_gdf': bairros,
}

with open(OUT_PATH, 'wb') as f:
    pickle.dump(data_pack, f)

print(f'Arquivo salvo: {OUT_PATH} | nodes: {num_nodes} | datas: {num_dates}')

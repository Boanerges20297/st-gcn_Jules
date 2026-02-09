#!/usr/bin/env python
"""
Expandir o grafo para incluir as 2374 comunidades com timeseries zerado
Mantém os 319 nós originais com seus dados e adiciona comunidades como nós "vazios"
"""

import pickle
import numpy as np
import pandas as pd
import os
from scipy.sparse import lil_matrix, csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')

print("=" * 80)
print("🔧 EXPANDINDO GRAFO PARA INCLUIR COMUNIDADES COM FEATURES")
print("=" * 80)

# Carregar dados atuais (com 2693 nós no nodes_gdf)
print("\n1. Carregando dados...")
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data['nodes_gdf']
node_features_original = data.get('node_features')  # (319, T, 26)
adj_geo_original = data.get('adj_geo')  # (319, 319)
adj_conflict_original = data.get('adj_conflict')  # (319, 319)
dates = data.get('dates')
feature_names = data.get('feature_names')

print(f"   nodes_gdf: {len(nodes_gdf)} nós")
print(f"   node_features: {node_features_original.shape}")
print(f"   adj_geo: {adj_geo_original.shape if adj_geo_original is not None else 'None'}")
print(f"   adj_conflict: {adj_conflict_original.shape if adj_conflict_original is not None else 'None'}")

# Identificar quais são os 319 originais (bairro + cidade) vs 2374 novos (comunidade)
original_nodes = nodes_gdf[nodes_gdf['node_type'].isin(['bairro', 'cidade'])]
community_nodes = nodes_gdf[nodes_gdf['node_type'] == 'comunidade']

print(f"\n2. Estrutura dos nós:")
print(f"   - Originais (bairro + cidade): {len(original_nodes)}")
print(f"   - Comunidades (novo): {len(community_nodes)}")
print(f"   - Total: {len(nodes_gdf)}")

# Expandir node_features
print(f"\n3. Expandindo node_features...")
num_timesteps = node_features_original.shape[1]
num_channels = node_features_original.shape[2]
num_total_nodes = len(nodes_gdf)

# Criar nova matriz (2693, T, 26) com zeros
node_features_expanded = np.zeros((num_total_nodes, num_timesteps, num_channels), dtype=np.float32)

# Copiar dados originais (319 primeiros nós)
node_features_expanded[:319, :, :] = node_features_original

print(f"   Dimensões: {node_features_original.shape} → {node_features_expanded.shape}")
print(f"   ✅ 319 nós originais: preenchidos com dados históricos")
print(f"   ✅ 2374 nós comunidade: preenchidos com zeros")

# Expandir adj_geo
print(f"\n4. Expandindo adj_geo...")
adj_geo_expanded = lil_matrix((num_total_nodes, num_total_nodes), dtype=np.float32)
adj_geo_expanded[:319, :319] = adj_geo_original

print(f"   Dimensões: {adj_geo_original.shape} → {adj_geo_expanded.shape}")
print(f"   ✅ Submatriz 0:319 preenchida")

# Expandir adj_conflict
print(f"\n5. Expandindo adj_conflict...")
adj_conflict_expanded = lil_matrix((num_total_nodes, num_total_nodes), dtype=np.float32)
adj_conflict_expanded[:319, :319] = adj_conflict_original

print(f"   Dimensões: {adj_conflict_original.shape} → {adj_conflict_expanded.shape}")
print(f"   ✅ Submatriz 0:319 preenchida")

# Conectar comunidades aos nós vizinhos geograficamente
print(f"\n6. Conectando comunidades aos vizinhos geográficos...")
from scipy.spatial import cKDTree

original_coords = original_nodes[['longitude', 'latitude']].values
community_coords = community_nodes[['longitude', 'latitude']].values

if len(community_coords) > 0 and len(original_coords) > 0:
    tree = cKDTree(original_coords)
    distances, indices = tree.query(community_coords, k=min(3, len(original_coords)))
    
    # Conectar cada comunidade aos seus 3 vizinhos mais próximos
    for i, community_idx in enumerate(range(319, 319 + len(community_nodes))):
        if isinstance(indices[i], (list, np.ndarray)):
            neighbors = indices[i]
            dists = distances[i]
        else:
            neighbors = [indices[i]]
            dists = [distances[i]]
        
        for neighbor_idx, dist in zip(neighbors, dists if isinstance(dists, (list, np.ndarray)) else [dists]):
            # Peso = 1 / distância (comunidades próximas têm conexão mais forte)
            if dist > 0:
                weight = 1.0 / (dist + 0.001)
            else:
                weight = 1.0
            
            # Usar adj_geo para conexões geográficas
            adj_geo_expanded[community_idx, neighbor_idx] = weight
            adj_geo_expanded[neighbor_idx, community_idx] = weight
    
    print(f"   ✅ {len(community_nodes)} comunidades conectadas aos vizinhos geográficos")

# Converter para CSR (format mais eficiente)
adj_geo_expanded = adj_geo_expanded.tocsr()
adj_conflict_expanded = adj_conflict_expanded.tocsr()

# Salvar dados atualizados
print(f"\n7. Salvando dados expandidos em pickle...")
data['node_features'] = node_features_expanded
data['adj_geo'] = adj_geo_expanded.toarray()
data['adj_conflict'] = adj_conflict_expanded.toarray()
data['nodes_gdf'] = nodes_gdf

with open(pickle_path, 'wb') as f:
    pickle.dump(data, f)

print(f"   ✅ Salvo em {pickle_path}")

# Verificação final
print(f"\n" + "=" * 80)
print("✅ GRAFO EXPANDIDO COM SUCESSO!")
print(f"=" * 80)
print(f"Novo estado:")
print(f"   - Nós: {len(nodes_gdf)} (319 orig + 2374 comunidades)")
print(f"   - Features: {node_features_expanded.shape} (T={num_timesteps}, F={num_channels})")
print(f"   - Adjacências: {adj_geo_expanded.shape} (geo), {adj_conflict_expanded.shape} (conflict)")
print(f"\n⚠️  IMPORTANTE:")
print(f"   - As 319 nós originais mantêm seus históricos")
print(f"   - As 2374 comunidades têm features zerados (sem histórico)")
print(f"   - Comunidades estarão visíveis no mapa com risco inicial = 0")
print(f"   - Podem ser afetadas por eventos exógenos")
print(f"=" * 80)

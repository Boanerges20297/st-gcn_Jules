#!/usr/bin/env python
"""
Expandir o grafo para incluir as 2374 comunidades com timeseries zerado
Mantém os 319 nós originais com seus dados e adiciona comunidades como nós "vazios"
mas com possibilidade de serem preenchidos com dados exógenos
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

if len(original_nodes) != node_features_original.shape[0]:
    print(f"\n⚠️  AVISO: Esperados {node_features_original.shape[0]}, mas encontrados {len(original_nodes)} nós originais")
    print("   Pode haver problemas de compatibilidade")

# Expandir node_features
print(f"\n3. Expandindo node_features...")
num_timesteps = node_features_original.shape[1]
num_channels = node_features_original.shape[2]
num_total_nodes = len(nodes_gdf)

# Criar nova matriz (2693, T, 26) com zeros
node_features_expanded = np.zeros((num_total_nodes, num_timesteps, num_channels), dtype=np.float32)

# Copiar dados originais (319 primeiros nós - assumindo que são os índices 0-318)
node_features_expanded[:319, :, :] = node_features_original

print(f"   Dimensões: {node_features_original.shape} → {node_features_expanded.shape}")
print(f"   ✅ 319 nós originais: preenchidos com dados históricos")
print(f"   ✅ 2374 nós comunidade: preenchidos com zeros (dados ausentes)")

# Expandir adjacency_matrix
print(f"\n4. Expandindo adjacency_matrix...")
adj_original_sp = csr_matrix(adj_matrix_original)
adj_expanded = lil_matrix((num_total_nodes, num_total_nodes), dtype=np.float32)

# Copiar a adjacência original na subbatriz 0:319, 0:319
adj_expanded[:319, :319] = adj_original_sp

# As comunidades não têm conexões inicialmente (podem vir de eventos exógenos)
# Mas podemos conectá-las aos bairros/cidades mais próximos geograficamente
print(f"   Conectando comunidades aos nós vizinhos geograficamente...")

from scipy.spatial import cKDTree

# Calcular distâncias entre comunidades e nós originais
original_coords = original_nodes[['longitude', 'latitude']].values
community_coords = community_nodes[['longitude', 'latitude']].values

if len(community_coords) > 0 and len(original_coords) > 0:
    tree = cKDTree(original_coords)
    distances, indices = tree.query(community_coords, k=min(3, len(original_coords)))
    
    # Conectar cada comunidade aos seus 3 vizinhos mais próximos
    for i, community_idx in enumerate(range(319, 319 + len(community_nodes))):
        if isinstance(indices[i], (list, np.ndarray)):
            neighbors = indices[i]
        else:
            neighbors = [indices[i]]
        
        for neighbor_idx in neighbors:
            # Peso da aresta = 1 / distância (comunidades próximas têm conexão mais forte)
            weight = 1.0 / max(distances[i] if isinstance(distances[i], (int, float)) else 1.0, 0.01)
            adj_expanded[community_idx, neighbor_idx] = weight
            adj_expanded[neighbor_idx, community_idx] = weight

adj_expanded = csr_matrix(adj_expanded)
print(f"   Dimensões: {adj_matrix_original.shape} → {adj_expanded.shape}")
print(f"   ✅ Submatriz 0:319 original preenchida")
print(f"   ✅ Comunidades conectadas aos nós vizinhos")

# Salvar dados atualizados
print(f"\n5. Atualizando pickle...")
data['node_features'] = node_features_expanded
data['adjacency_matrix'] = adj_expanded.toarray()  # Converter para array denso
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
print(f"   - Adjacência: {adj_expanded.shape}")
print(f"\n⚠️  IMPORTANTE:")
print(f"   - As 319 nós originais mantêm seus históricos (26 canais)")
print(f"   - As 2374 comunidades têm features zerados (sem histórico)")
print(f"   - Comunidades estarão visíveis no mapa com risco inicial = 0")
print(f"   - Podem ser afetadas por eventos exógenos")
print(f"=" * 80)

import pickle
import pandas as pd
import numpy as np

# Load nodes GDF
with open('data/processed/graph_data/nodes_gdf.pkl', 'rb') as f:
    gdf = pickle.load(f)

# Load processed graph data
with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    graph_data = pickle.load(f)

print('Analysis of node mapping:')
print(f'Total nodes in GDF: {len(gdf)}')
print(f'Nodes used in graph: 1491')
print(f'Difference: {len(gdf) - 1491}')
print()

# Check adjacency matrix to understand which nodes are active
adj_geo = graph_data['adj_geo']
print(f'Adjacency matrix shape: {adj_geo.shape}')

# Count active nodes (have at least one connection)
if hasattr(adj_geo, 'sum'):  # if sparse matrix
    active_nodes = (adj_geo.sum(axis=0) > 0).A1.sum()
else:
    active_nodes = (adj_geo.sum(axis=0) > 0).sum()

print(f'Active nodes in adjacency matrix: {active_nodes}')
print()

# Create mapping: node_index (0-1490) -> node_name from gdf
print('Creating node index mapping...')
# Since we have 1491 nodes in graph and 2378 in GDF, 
# it appears graph only uses first 1491 nodes
node_mapping = {}
for i in range(1491):
    if i < len(gdf):
        node_mapping[i] = gdf.iloc[i]['name']
    else:
        node_mapping[i] = f'Node_{i}'

print(f'Created mapping for {len(node_mapping)} nodes')
print()
print('Sample mapping:')
for i in range(0, 10):
    print(f'  node {i}: {node_mapping[i]}')
print('  ...')
for i in range(1485, 1491):
    print(f'  node {i}: {node_mapping[i]}')

# Save mapping
with open('data/processed/node_index_to_name_mapping.pkl', 'wb') as f:
    pickle.dump(node_mapping, f)
print()
print('Saved mapping to data/processed/node_index_to_name_mapping.pkl')

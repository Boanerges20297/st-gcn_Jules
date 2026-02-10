import pickle
import pandas as pd

# Carregar nodes_gdf.pkl diretamente
nodes = pickle.load(open('data/processed/graph_data/nodes_gdf.pkl','rb'))

print(f'Shape: {nodes.shape}')
print(f'Columns: {list(nodes.columns)}')
if 'node_type' in nodes.columns:
    print(f'Node types: {nodes["node_type"].unique()}')
print(f'Geom types: {nodes.geometry.geom_type.unique()}')
print(f'\nFirst 3 rows:')
for idx in range(min(3, len(nodes))):
    row = nodes.iloc[idx]
    print(f"  Index {idx}: {row['name']}, type={row.get('node_type')}, geom={row.geometry.geom_type}")


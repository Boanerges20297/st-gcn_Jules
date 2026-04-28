import pickle
import os

ROOT_DIR = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
path = os.path.join(ROOT_DIR, 'data', 'processed', 'processed_fortaleza.pkl')

with open(path, 'rb') as f:
    data = pickle.load(f)

if 'nodes_gdf' in data and 'name' in data['nodes_gdf'].columns:
    node_names = data['nodes_gdf']['name'].tolist()
elif 'nodes' in data:
    node_names = data['nodes']
else:
    node_names = []

indices = [71, 25, 54, 60, 20, 0]
for idx in indices:
    if idx < len(node_names):
        print(f"Index {idx}: {node_names[idx]}")
    else:
        print(f"Index {idx}: OUT OF RANGE")

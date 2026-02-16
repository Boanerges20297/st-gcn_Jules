import pickle

with open('data/processed/processed_rmf.pkl', 'rb') as f:
    data = pickle.load(f)

nodes = data['nodes_gdf']['name'].tolist()
nodes.sort()
print(f"RMF Nodes in pkl ({len(nodes)}):")
for n in nodes:
    print(f"  - {n}")

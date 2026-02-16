import pickle
import pandas as pd
import numpy as np

with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data['nodes_gdf']
node_features = data['node_features']

# Filter interior
interior_mask = nodes_gdf['regiao'] == 'interior'
indices = np.where(interior_mask)[0]
interior_nodes = nodes_gdf.iloc[indices]
interior_features = node_features[indices, :, :]

# Count total CVLI (Channel 0) for each node
activity = interior_features[:, :, 0].sum(axis=1)
interior_nodes = interior_nodes.copy()
interior_nodes['activity'] = activity

# Sort by activity
inactive = interior_nodes.sort_values('activity').head(10)
print("Least active Interior nodes:")
for idx, row in inactive.iterrows():
    print(f"{row['name']}: {row['activity']}")

print("\nTotal Interior nodes in pkl: " + str(len(interior_nodes)))

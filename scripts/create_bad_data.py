import pickle
import numpy as np
import os

# Create mock BAD data (2378 nodes, 2 channels)
# This simulates the state of the user's machine
def create_bad_data():
    num_nodes = 2378
    num_timesteps = 100
    num_channels = 2

    # Fake features
    node_features = np.random.rand(num_nodes, num_timesteps, num_channels).astype(np.float32)

    # Fake metadata needed by app.py
    data_pack = {
        'node_features': node_features,
        'adj_matrix': np.zeros((num_nodes, num_nodes)),
        'dates': range(num_timesteps),
        'nodes_gdf': None # Simplification, app handles None gdf
    }

    path = 'data/processed/processed_graph_data_BAD.pkl'
    with open(path, 'wb') as f:
        pickle.dump(data_pack, f)

    print(f"Created BAD data at {path}")

if __name__ == "__main__":
    create_bad_data()

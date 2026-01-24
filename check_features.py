import pickle
import numpy as np
import os

DATA_FILE = 'data/processed/processed_graph_data.pkl'

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    node_features = data['node_features']
    # node_features shape: (Nodes, Timesteps, Features)
    print(f"Shape: {node_features.shape}")

    # Feature 0 is CVLI
    cvli = node_features[:, :, 0]
    total_cvli = np.sum(cvli)
    max_cvli = np.max(cvli)
    nonzero_nodes = np.count_nonzero(np.sum(cvli, axis=1))

    print(f"Total CVLI events: {total_cvli}")
    print(f"Max daily CVLI in a node: {max_cvli}")
    print(f"Nodes with at least one CVLI: {nonzero_nodes} / {node_features.shape[0]}")

    # Check last 180 days
    last_180 = cvli[:, -180:]
    print(f"CVLI events in last 180 days: {np.sum(last_180)}")
    print(f"Nodes active in last 180 days: {np.count_nonzero(np.sum(last_180, axis=1))}")

else:
    print("Data file not found.")

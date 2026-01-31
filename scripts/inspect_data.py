import pickle
import numpy as np
import sys
import os

def inspect():
    path = 'data/processed/processed_graph_data.pkl'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    features = data['node_features']
    print(f"Feature Shape: {features.shape}")

    # Check shape: (Nodes, Days, 3)
    if len(features.shape) != 3 or features.shape[2] != 3:
        print("FAIL: Feature shape is not (Nodes, Days, 3)")
        sys.exit(1)

    # Check Channels
    cvli = features[:, :, 0]
    cvp = features[:, :, 1]
    tension = features[:, :, 2]

    print(f"CVLI Sum: {np.sum(cvli)}")
    print(f"CVP Sum: {np.sum(cvp)}")
    print(f"Tension Mean: {np.mean(tension)}")
    print(f"Tension Unique Values: {np.unique(tension)}")

    if np.sum(cvli) == 0:
        print("WARNING: CVLI channel is empty")

    if np.sum(cvp) == 0:
        print("WARNING: CVP channel is empty")

    if np.all(tension == 0):
        print("WARNING: Tension channel is all zeros")

    print("PASS: Data structure looks correct.")

if __name__ == "__main__":
    inspect()

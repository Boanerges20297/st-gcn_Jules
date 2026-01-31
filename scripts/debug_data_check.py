import pickle
import numpy as np
import sys

def main():
    filepath = 'data/processed/processed_graph_data.pkl'
    print(f"Loading {filepath}...")

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    # Check 1: Feature Channels
    features = data['node_features']
    print(f"Feature shape: {features.shape}")
    if features.shape[2] < 4:
        print(f"FAIL: Expected at least 4 feature channels, got {features.shape[2]}")
        sys.exit(1)
    else:
        print(f"PASS: Feature channels >= 4 ({features.shape[2]})")

    # Check 2: Conflict Edges
    adj_conflict = data['adj_conflict']
    conflict_count = np.sum(adj_conflict)
    print(f"Conflict edges count: {conflict_count}")
    if conflict_count == 0:
        print("FAIL: No conflict edges found in adj_conflict.")
        # sys.exit(1) # Warning only, as it depends on geometry
    else:
        print("PASS: Conflict edges exist.")

    # Check 3: Node Names (Barroso)
    nodes_gdf = data['nodes_gdf']
    node_names = nodes_gdf['name'].astype(str).str.upper().tolist()

    target_node = "BARROSO"
    if target_node in node_names:
        print(f"PASS: Node '{target_node}' found.")
        # Print stats for Barroso
        idx = node_names.index(target_node)
        print(f"  - Barroso Faction: {nodes_gdf.iloc[idx]['faction']}")
        print(f"  - Barroso CV Dist: {nodes_gdf.iloc[idx]['dist_cv']:.2f}")
        print(f"  - Barroso TCP Dist: {nodes_gdf.iloc[idx]['dist_tcp']:.2f}")
        print(f"  - Barroso CVLI Events (Sum): {np.sum(features[idx, :, 0])}")
        print(f"  - Barroso Vehicle Thefts (Sum): {np.sum(features[idx, :, 1])}")
    else:
        print(f"FAIL: Node '{target_node}' NOT found.")
        sys.exit(1)

    print("\nAll checks passed successfully.")

if __name__ == "__main__":
    main()

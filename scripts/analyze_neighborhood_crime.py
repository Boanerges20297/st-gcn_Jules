import pickle
import numpy as np
import pandas as pd
import os
import sys

# Adicionar raiz ao path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    path = os.path.join(ROOT, 'data', 'processed', 'processed_fortaleza.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    data = load_data()
    features = data['node_features'] # (Nodes, Time, Channels)
    nodes_gdf = data['nodes_gdf']
    dates = pd.to_datetime(data['dates'])

    # Calculate total CVLI per node
    # Channel 0 is CVLI
    total_cvli_per_node = features[:, :, 0].sum(axis=1) # (Nodes,)

    # Calculate months duration
    num_days = len(dates)
    num_months = num_days / 30.0

    print(f"Total Period: {num_days} days (~{num_months:.1f} months)")

    # Threshold: <= 1 per month => total <= num_months
    threshold_total = num_months * 1.0

    print(f"Threshold for exclusion (<= 1/month): {threshold_total:.1f} total crimes")

    active_mask = total_cvli_per_node > threshold_total
    active_indices = np.where(active_mask)[0]
    excluded_indices = np.where(~active_mask)[0]

    print(f"\nTotal Nodes: {len(nodes_gdf)}")
    print(f"Active Nodes (> 1/mo): {len(active_indices)}")
    print(f"Excluded Nodes (<= 1/mo): {len(excluded_indices)}")

    print("\n--- EXCLUDED NEIGHBORHOODS (NOISE/SAFE) ---")
    excluded_names = nodes_gdf.iloc[excluded_indices]['name'].values
    for name in np.sort(excluded_names):
        idx = nodes_gdf[nodes_gdf['name'] == name].index[0]
        count = total_cvli_per_node[idx]
        print(f"{name}: {count:.0f} crimes ({count/num_months:.2f}/mo)")

    print("\n--- ACTIVE NEIGHBORHOODS (TRAINING TARGETS) ---")
    active_names = nodes_gdf.iloc[active_indices]['name'].values
    print(f"Top 10 most active: {active_names[:10]}...")

if __name__ == "__main__":
    main()

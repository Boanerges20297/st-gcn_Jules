import pickle
import os
import numpy as np
import sys

# Add root to path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def test_window_viability():
    print("Loading data...")
    data_file = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')

    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    with open(data_file, 'rb') as f:
        pack = pickle.load(f)

    node_features = pack['node_features']
    dates = pack['dates']

    print(f"Original Data Shape: {node_features.shape}")
    print(f"Original Date Range: {dates[0]} to {dates[-1]}")

    # 1. Filter for 2025
    start_date_str = '2025-01-01'
    start_idx = -1

    for idx, d in enumerate(dates):
        if str(d) >= start_date_str:
            start_idx = idx
            break

    if start_idx == -1:
        print("No data found for >= 2025-01-01")
        return

    filtered_features = node_features[:, start_idx:, :]
    filtered_dates = dates[start_idx:]

    print(f"\nFiltered Data (>= {start_date_str}):")
    print(f"Shape: {filtered_features.shape}")
    print(f"Date Range: {filtered_dates[0]} to {filtered_dates[-1]}")

    num_timesteps = filtered_features.shape[1]

    # 2. Check Window Viability
    # CVLI: 90 input, 7 output
    history_window = 90
    horizon = 7

    valid_range = num_timesteps - history_window - horizon + 1

    print(f"\nTesting Window Config: Input={history_window}, Output={horizon}")
    print(f"Required consecutive days: {history_window + horizon}")
    print(f"Available days: {num_timesteps}")

    if valid_range <= 0:
        print(f"FAIL: Not enough data. Need {history_window + horizon}, have {num_timesteps}.")
        sys.exit(1)
    else:
        print(f"SUCCESS: Sufficient data available.")
        print(f"Resulting Samples (Batches): {valid_range}")

    # Optional: Simulate creation (lightweight)
    print("\nSimulating sample generation (metadata only)...")
    samples = 0
    for i in range(0, valid_range):
        samples += 1

    print(f"Total training samples that would be generated: {samples}")

if __name__ == "__main__":
    test_window_viability()

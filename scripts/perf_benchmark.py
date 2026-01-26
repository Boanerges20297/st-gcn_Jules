import sys
import os
import time
import numpy as np
import torch
import pickle

# Add root to sys.path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Importing app (this triggers initial load)...")
start_import = time.time()
import app
print(f"Import took {time.time() - start_import:.4f}s")

def benchmark_baseline():
    print("\n--- Benchmarking Baseline (Full Reload) ---")
    start = time.time()

    # Simulate what happens in save_exogenous currently:
    # It calls load_data_and_models()
    app.load_data_and_models()

    end = time.time()
    print(f"Baseline Full Reload took: {end - start:.4f}s")
    return end - start

def benchmark_optimized_actual():
    print("\n--- Benchmarking Optimized Actual (update_exogenous_state) ---")

    # We verify that original_adj_matrix is set
    if app.original_adj_matrix is None:
        print("Error: original_adj_matrix is None!")
        return 0

    start = time.time()

    # Call the actual new function
    app.update_exogenous_state()

    end = time.time()
    print(f"Optimized Update took: {end - start:.4f}s")
    return end - start

if __name__ == "__main__":
    # Run Optimized first to ensure it works
    opt_time = benchmark_optimized_actual()

    # Run Baseline (this will reload everything and reset state, which is fine)
    baseline_time = benchmark_baseline()

    improvement = baseline_time / opt_time if opt_time > 0 else 0
    print(f"\nSpeedup Factor: {improvement:.2f}x")

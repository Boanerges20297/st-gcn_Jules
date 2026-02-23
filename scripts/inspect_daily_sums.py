import pickle
import numpy as np
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
    features = data['node_features']

    # Calculate daily CVLI sums
    daily_sums = features[:, :, 0].sum(axis=0) # (TimeSteps,)

    print(f"Total days: {len(daily_sums)}")
    print(f"Min: {daily_sums.min()}")
    print(f"Max: {daily_sums.max()}")
    print(f"Mean: {daily_sums.mean()}")
    print(f"Std: {daily_sums.std()}")
    print(f"25th percentile: {np.percentile(daily_sums, 25)}")
    print(f"50th percentile (Median): {np.median(daily_sums)}")
    print(f"75th percentile: {np.percentile(daily_sums, 75)}")
    print(f"90th percentile: {np.percentile(daily_sums, 90)}")
    print(f"95th percentile: {np.percentile(daily_sums, 95)}")
    print(f"Days with > 10 CVLI: {np.sum(daily_sums > 10)}")
    print(f"Days with > 5 CVLI: {np.sum(daily_sums > 5)}")
    print(f"Days with > 3 CVLI: {np.sum(daily_sums > 3)}")

if __name__ == "__main__":
    main()

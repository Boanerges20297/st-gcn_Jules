import pickle
import os
import sys

# Mock geopandas to avoid ModuleNotFoundError when loading the pickle
class MockGeoPandas:
    def __getattr__(self, name):
        return None

sys.modules['geopandas'] = MockGeoPandas()

def check_dates():
    path = 'data/processed/processed_fortaleza.pkl'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    with open(path, 'rb') as f:
        # Some pickles might fail if they require real geopandas structures for unpickling
        try:
            data = pickle.load(f)
            dates = data.get('dates', [])
            print(f"Total days: {len(dates)}")
            if len(dates) > 0:
                print(f"First: {dates[0]}")
                print(f"Last: {dates[-1]}")
                if len(dates) >= 120:
                    print(f"Window (last 120 steps) start: {dates[-120]}")
                else:
                    print(f"Warning: Only {len(dates)} days available.")
            
            nf = data.get('node_features')
            if nf is not None:
                print(f"Node Features shape: {nf.shape}")
                # The orchestrator uses: x_raw = data['node_features'][:, -window:, :].copy()
                # with window = 120
                if nf.shape[1] >= 120:
                    print(f"CONFIRMED: History of 120 days is available in the tensor.")
                else:
                    print(f"ERROR: Tensor has only {nf.shape[1]} days.")
        except Exception as e:
            print(f"Failed to load pickle: {e}")

if __name__ == '__main__':
    check_dates()

import pickle
import os
import sys
import numpy as np

# Mock geopandas
class MockGeoPandas:
    class GeoDataFrame: pass
    def __getattr__(self, name): return None
sys.modules['geopandas'] = MockGeoPandas()

def debug_efficiency():
    path = 'data/processed/processed_fortaleza.pkl'
    if not os.path.exists(path):
        print("File not found.")
        return
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
        nf = data.get('node_features')
        dates = data.get('dates', [])
        
        print(f"Total dates: {len(dates)}")
        print(f"Last date in dataset: {dates[-1]}")
        
        # Check CVLI channel (0) for the last 7 days
        recent_cvli = nf[:, -7:, 0]
        total_cvli = recent_cvli.sum()
        print(f"Total CVLI in last 7 days of dataset: {total_cvli}")
        
        if total_cvli > 0:
            # Show top nodes with crimes
            node_sums = recent_cvli.sum(axis=1)
            top_indices = np.argsort(node_sums)[-5:][::-1]
            print("Top nodes with crimes in last 7 days:")
            for idx in top_indices:
                name = data['nodes_gdf'].iloc[idx]['name']
                print(f" - {name}: {node_sums[idx]} crimes")
        else:
            print("WARNING: No CVLI found in the last 7 steps of node_features.")
            # Check more steps back
            all_cvli = nf[:, :, 0].sum()
            print(f"Total CVLI in whole dataset: {all_cvli}")

if __name__ == '__main__':
    debug_efficiency()

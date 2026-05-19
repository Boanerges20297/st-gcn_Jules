import pickle
import pandas as pd
import numpy as np

def debug():
    # Load processed_fortaleza.pkl
    path = 'data/processed/processed_fortaleza.pkl'
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        print("Keys in processed_fortaleza.pkl:", data.keys())
        nodes_gdf = data.get("nodes_gdf")
        if nodes_gdf is not None:
            print("Total nodes:", len(nodes_gdf))
            # Find Canindezinho
            can_nodes = nodes_gdf[nodes_gdf['name'].str.contains('caninde', case=False, na=False)]
            print("\nNodes matching 'caninde':")
            print(can_nodes[['name', 'regiao', 'geometry']])
            
            # Print node at index 275
            if 275 < len(nodes_gdf):
                print(f"\nNode at index 275:")
                print(nodes_gdf.iloc[275])
            else:
                print(f"\nIndex 275 out of range (max index is {len(nodes_gdf)-1})")
    except Exception as e:
        print("Error loading fortaleza:", e)

    # Let's also check other pkl files if any
    import glob
    for p in glob.glob('data/processed/processed_*.pkl'):
        if 'fortaleza' in p: continue
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            ngdf = d.get("nodes_gdf")
            if ngdf is not None:
                can_nodes = ngdf[ngdf['name'].str.contains('caninde', case=False, na=False)]
                if len(can_nodes) > 0:
                    print(f"\nFound 'caninde' in {p}:")
                    print(can_nodes[['name', 'regiao', 'geometry']])
        except Exception as e:
            print(f"Error loading {p}:", e)

if __name__ == '__main__':
    debug()

import json
import os
import pickle

BASE_DIR = os.path.dirname(__file__)
pkl_path = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.pkl')

try:
    with open(pkl_path, 'rb') as f:
        ngdf = pickle.load(f)
        print(f"Total nodes: {len(ngdf)}")
        print(f"\nNode types distribution:")
        if 'node_type' in ngdf.columns:
            types = ngdf['node_type'].value_counts()
            print(types)
        else:
            print("No node_type column found")
        
        print(f"\nFirst 10 nodes:")
        for idx, row in ngdf.head(10).iterrows():
            print(f"{idx}: {row.get('name', 'N/A')} - type={row.get('node_type', 'N/A')}")
except Exception as e:
    print(f"ERROR: {e}")

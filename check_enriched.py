import json
import os
import pickle

BASE_DIR = os.path.dirname(__file__)

# Check nodes_gdf_enriched.pkl
enriched_path = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf_enriched.pkl')

try:
    with open(enriched_path, 'rb') as f:
        ngdf_enriched = pickle.load(f)
        print(f"nodes_gdf_enriched loaded: {len(ngdf_enriched)} nodes")
        print(f"Columns: {list(ngdf_enriched.columns)}")
        
        if 'node_type' in ngdf_enriched.columns:
            types = ngdf_enriched['node_type'].value_counts()
            print(f"\nNode types:")
            print(types)
        
        # Show first few rows
        print(f"\nFirst 5 nodes:")
        for idx, row in ngdf_enriched.head(5).iterrows():
            print(f"  {row.get('name', 'N/A')} - type={row.get('node_type', 'N/A')}")
        
        # Check if there are any comunidades/favelas
        if 'node_type' in ngdf_enriched.columns:
            dynamic = ngdf_enriched[ngdf_enriched['node_type'] == 'dynamic_node']
            print(f"\nDynamic nodes: {len(dynamic)}")
            if len(dynamic) > 0:
                print(dynamic.head())
except Exception as e:
    print(f"ERROR: {e}")

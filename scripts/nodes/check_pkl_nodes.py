import pickle
import pandas as pd

with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data.get('nodes_gdf')
if nodes_gdf is not None:
    print(f"Total nodes in pkl: {len(nodes_gdf)}")
    if 'regiao' in nodes_gdf.columns:
        print("Counts by region:")
        print(nodes_gdf['regiao'].value_counts())
    elif 'region_type' in nodes_gdf.columns:
        print("Counts by region_type:")
        print(nodes_gdf['region_type'].value_counts())
    
    # List some nodes
    print("\nFirst 10 nodes:")
    print(nodes_gdf['name'].head(10).tolist())
    
    # Check if there are Interior nodes
    interior = nodes_gdf[nodes_gdf.get('regiao', '') == 'fortaleza']
    print(f"Fortaleza nodes: {len(interior)}")
else:
    print("nodes_gdf not found in pkl")

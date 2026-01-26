import pickle
import pandas as pd
import geopandas as gpd
import os

DATA_FILE = 'data/processed/processed_graph_data.pkl'

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    nodes_gdf = data.get('nodes_gdf')
    if nodes_gdf is not None:
        print("Total Nodes:", len(nodes_gdf))
        if 'CIDADE' in nodes_gdf.columns:
            print("\nCIDADE Value Counts:")
            print(nodes_gdf['CIDADE'].value_counts())

            # Check for generic 'RMF/Interior' or Empty
            missing = nodes_gdf[nodes_gdf['CIDADE'].isna() | (nodes_gdf['CIDADE'] == '') | (nodes_gdf['CIDADE'] == 'RMF/Interior')]
            print(f"\nMissing/Generic CIDADE: {len(missing)}")
        else:
            print("\nColumn 'CIDADE' NOT found in nodes_gdf.")
    else:
        print("nodes_gdf not found in pickle.")
else:
    print(f"File not found: {DATA_FILE}")

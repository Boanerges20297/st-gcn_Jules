import pickle
import pandas as pd
import os

DATA_FILE = 'data/processed/processed_graph_data.pkl'

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    nodes_gdf = data['nodes_gdf']
    print("Columns:", nodes_gdf.columns)
    print("First 5 rows:")
    print(nodes_gdf[['name', 'faction'] if 'faction' in nodes_gdf.columns else ['name']].head())

    if 'regiao' in nodes_gdf.columns: # Check for variations
        print("Regiao column found.")
    else:
        print("Regiao column NOT found.")
        print("Available columns:", nodes_gdf.columns.tolist())

else:
    print("Data file not found.")

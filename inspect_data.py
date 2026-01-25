import pickle
import geopandas as gpd
import pandas as pd
import os

DATA_FILE = 'data/processed/processed_graph_data.pkl'

if not os.path.exists(DATA_FILE):
    print(f"File not found: {DATA_FILE}")
else:
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
        nodes = data['nodes_gdf']
        print(f"Total Nodes: {len(nodes)}")
        print("Columns:", nodes.columns)

        if 'geometry' in nodes.columns:
            print("Bounds:", nodes.total_bounds)

        if 'CIDADE' in nodes.columns:
            print("Cities:", nodes['CIDADE'].unique())
        else:
            print("No CIDADE column.")

        print("Sample Names:", nodes['name'].head(10).tolist())

        # Check if there are nodes outside Fortaleza bbox roughly
        # Fortaleza approx: -38.6, -3.9 to -38.4, -3.65
        outliers = nodes.cx[-38.4:, :]
        print(f"Nodes East of -38.4: {len(outliers)}")
        outliers_west = nodes.cx[: -38.66, :]
        print(f"Nodes West of -38.66: {len(outliers_west)}")

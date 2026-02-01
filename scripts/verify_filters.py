import pickle
import pandas as pd
import geopandas as gpd
import os

DATA_FILE = 'data/processed/processed_graph_data.pkl'

def verify():
    if not os.path.exists(DATA_FILE):
        print(f"File not found: {DATA_FILE}")
        return

    print(f"Loading {DATA_FILE}...")
    try:
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)

        nodes_gdf = data['nodes_gdf']
        print(f"Loaded {len(nodes_gdf)} nodes.")

        if 'regiao' in nodes_gdf.columns:
            print("✓ Column 'regiao' exists.")
            print("\nValue Counts for 'regiao':")
            print(nodes_gdf['regiao'].value_counts())

            # Check for mapping issues
            unique_vals = nodes_gdf['regiao'].unique()
            # 'fortaleza' is expected in raw data, app.py handles normalization

            # Simulate app.py logic
            nodes_gdf['region_type'] = nodes_gdf['regiao'].replace('fortaleza', 'capital')

            # Simulate Filtering
            regions = ['capital', 'rmf', 'interior']

            for r in regions:
                print(f"\n--- Filter: {r} ---")
                filtered = nodes_gdf[nodes_gdf['region_type'] == r]
                print(f"Count: {len(filtered)}")
                if len(filtered) > 0:
                    print(f"Sample names: {filtered['name'].head(5).tolist()}")
                else:
                    print("No nodes found for this filter.")

        else:
            print("✗ Column 'regiao' MISSING!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify()

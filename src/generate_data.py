import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
from datetime import datetime

def generate_dummy_data():
    if not os.path.exists('data'):
        os.makedirs('data')

    # Create 30 dummy nodes to cover regions
    num_nodes = 30
    polygons = []
    regions = []

    # Generate nodes distributed in 3 regions
    for i in range(num_nodes):
        if i < 10:
            region = 'Fortaleza'
            base_lat, base_lon = -3.73, -38.52
        elif i < 20:
            region = 'RMF'
            base_lat, base_lon = -3.80, -38.60
        else:
            region = 'Interior'
            base_lat, base_lon = -4.00, -39.00

        lat = base_lat + (i % 10 * 0.01)
        lon = base_lon + (i % 10 * 0.01)

        poly = Polygon([
            (lon, lat),
            (lon + 0.008, lat),
            (lon + 0.008, lat + 0.008),
            (lon, lat + 0.008),
            (lon, lat)
        ])
        polygons.append(poly)
        regions.append(region)

    nodes_gdf = gpd.GeoDataFrame({
        'node_id': range(num_nodes),
        'faction': np.random.choice(['CV', 'PCC', 'Neutro', 'GDE'], num_nodes),
        'region': regions,
        'geometry': polygons
    }, crs="EPSG:4326")

    # Adjacency matrix (connect neighbors locally within regions)
    adj_matrix = np.eye(num_nodes)
    for i in range(num_nodes - 1):
        if regions[i] == regions[i+1]: # Only connect if same region for this dummy logic
            adj_matrix[i, i+1] = 1
            adj_matrix[i+1, i] = 1

    # Node features: (Num Nodes, Num Timesteps, Num Features)
    # Generate data until Today to allow date filtering
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=730) # 2 years
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    num_timesteps = len(dates)
    num_features = 2 # CVLI, CVP

    # Create some patterns
    node_features = np.random.randint(0, 3, size=(num_nodes, num_timesteps, num_features))

    # Add spikes to random nodes
    for _ in range(50):
        n = np.random.randint(0, num_nodes)
        t = np.random.randint(0, num_timesteps)
        node_features[n, t, 0] += 5 # High CVLI

    data_pack = {
        'node_features': node_features,
        'adj_matrix': adj_matrix,
        'nodes_gdf': nodes_gdf,
        'dates': dates
    }

    with open('data/processed_graph_data.pkl', 'wb') as f:
        pickle.dump(data_pack, f)

    print(f"Dummy data generated: {num_nodes} nodes, {num_timesteps} days.")

if __name__ == "__main__":
    generate_dummy_data()

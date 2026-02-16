import pickle
import os
import glob

files = glob.glob('data/processed/*.pkl')
for f_path in files:
    try:
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
        nodes_gdf = data.get('nodes_gdf')
        if nodes_gdf is not None:
            print(f"{f_path}: {len(nodes_gdf)} nodes")
        else:
            print(f"{f_path}: nodes_gdf not found")
    except Exception as e:
        print(f"{f_path}: Error {e}")

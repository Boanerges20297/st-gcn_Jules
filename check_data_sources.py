import json
import os

BASE_DIR = os.path.dirname(__file__)

# Test AIS - CAPITAL.geojson
ais_capital_path = os.path.join(BASE_DIR, 'data', 'static', 'AIS - CAPITAL.geojson')

try:
    with open(ais_capital_path, 'r', encoding='utf-8') as f:
        ais_data = json.load(f)
        features = ais_data.get('features', [])
        print(f"AIS features loaded: {len(features)}")
        
        if len(features) > 0:
            props = features[0].get('properties', {})
            print(f"Properties keys: {list(props.keys())}")
            print(f"First feature: {props}")
except Exception as e:
    print(f"ERROR: {e}")
    
# Test micro-nós pickle
pkl_path = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.pkl')
import pickle
try:
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            ngdf = pickle.load(f)
            print(f"\nMicro-nodes loaded: {len(ngdf)}")
            if len(ngdf) > 0:
                print(f"First micro-node:\n{ngdf.iloc[0]}")
    else:
        print(f"\nnodes_gdf.pkl not found: {pkl_path}")
except Exception as e:
    print(f"ERROR loading micro-nodes: {e}")

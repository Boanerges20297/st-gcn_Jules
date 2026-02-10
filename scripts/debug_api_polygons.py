#!/usr/bin/env python3
import os
import json
import sys
sys.path.insert(0, os.path.dirname(__file__))

from app import app, nodes_gdf
import pickle

BASE_DIR = os.path.dirname(__file__)

# Test 1: Check if AIS - CAPITAL.geojson exists and loads
ais_capital_path = os.path.join(BASE_DIR, 'data', 'static', 'AIS - CAPITAL.geojson')
print(f"[TEST 1] AIS - CAPITAL.geojson exists: {os.path.exists(ais_capital_path)}")

if os.path.exists(ais_capital_path):
    try:
        with open(ais_capital_path, 'r', encoding='utf-8') as f:
            ais_data = json.load(f)
            print(f"[TEST 1] Loaded {len(ais_data.get('features', []))} polygon features from AIS")
            if len(ais_data.get('features', [])) > 0:
                print(f"[TEST 1] First polygon: {ais_data['features'][0]['properties'].get('name', 'N/A')}")
    except Exception as e:
        print(f"[TEST 1] ERROR: {e}")

# Test 2: Check if nodes_gdf is loaded
print(f"\n[TEST 2] nodes_gdf global variable: {nodes_gdf is not None}")
if nodes_gdf is not None:
    print(f"[TEST 2] nodes_gdf shape: {nodes_gdf.shape}")
    print(f"[TEST 2] nodes_gdf columns: {list(nodes_gdf.columns)}")
    if len(nodes_gdf) > 0:
        print(f"[TEST 2] First node: {nodes_gdf.iloc[0].get('name', 'N/A')}")

# Test 3: Check if nodes_gdf.pkl exists
pkl_path = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.pkl')
print(f"\n[TEST 3] nodes_gdf.pkl exists: {os.path.exists(pkl_path)}")
if os.path.exists(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            ngdf = pickle.load(f)
            print(f"[TEST 3] Loaded {len(ngdf)} micro-nodes from pickle")
            if len(ngdf) > 0:
                print(f"[TEST 3] First micro-node: {ngdf.iloc[0].get('name', 'N/A')}")
                print(f"[TEST 3] First micro-node geometry: {ngdf.iloc[0].geometry}")
    except Exception as e:
        print(f"[TEST 3] ERROR: {e}")

# Test 4: Call /api/polygons endpoint
print(f"\n[TEST 4] Calling /api/polygons endpoint...")
with app.test_client() as client:
    response = client.get('/api/polygons')
    print(f"[TEST 4] Status: {response.status_code}")
    data = response.get_json()
    if data:
        features = data.get('features', [])
        print(f"[TEST 4] Returned {len(features)} features")
        if len(features) > 0:
            for i, feat in enumerate(features[:3]):
                geom_type = feat.get('geometry', {}).get('type', 'unknown')
                name = feat.get('properties', {}).get('name', 'N/A')
                print(f"[TEST 4] Feature {i}: {geom_type} - {name}")
    else:
        print(f"[TEST 4] ERROR: No JSON returned or parse error")
        print(f"[TEST 4] Response text: {response.data[:500]}")

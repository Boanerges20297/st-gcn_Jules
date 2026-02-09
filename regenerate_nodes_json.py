#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate nodes_gdf.json from pickle with correct encoding."""

import pickle
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nodes_pkl = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.pkl')
nodes_json_out = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf_regenerated.json')

print(f"Loading pickle from: {nodes_pkl}")

try:
    with open(nodes_pkl, 'rb') as f:
        nodes_gdf = pickle.load(f)
    
    print(f"Loaded {len(nodes_gdf)} nodes")
    print(f"Type: {type(nodes_gdf)}")
    
    if hasattr(nodes_gdf, 'to_dict'):
        # Convert GeoDataFrame to dict format
        nodes_dict = {}
        for idx, row in nodes_gdf.iterrows():
            node_id = str(int(idx)) if isinstance(idx, (int, float)) else str(idx)
            node_data = row.to_dict()
            # Remove geometry object, keep coordinates
            if 'geometry' in node_data and hasattr(node_data['geometry'], 'x'):
                geom = node_data['geometry']
                node_data['latitude'] = float(geom.y)
                node_data['longitude'] = float(geom.x)
                del node_data['geometry']
            nodes_dict[node_id] = node_data
        
        # Save with proper UTF-8 encoding
        with open(nodes_json_out, 'w', encoding='utf-8') as f:
            json.dump(nodes_dict, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✓ Saved to: {nodes_json_out}")
        print(f"  Size: {os.path.getsize(nodes_json_out)} bytes")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

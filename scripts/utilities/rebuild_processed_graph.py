#!/usr/bin/env python3
"""Rebuild a single processed_graph_data.pkl from the chunked files
located in data/processed/graph_data/. This is intended for local use only;
the resulting file is added to .gitignore to avoid committing large blobs.
"""
import os
import json
import pickle
import numpy as np
import geopandas as gpd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GRAPH_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
OUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')


def load_chunks(graph_dir):
    pack = {}
    if not os.path.isdir(graph_dir):
        raise FileNotFoundError(f'Graph dir not found: {graph_dir}')

    # nodes_gdf
    ng_pkl = os.path.join(graph_dir, 'nodes_gdf.pkl')
    ng_json = os.path.join(graph_dir, 'nodes_gdf.json')
    if os.path.exists(ng_pkl):
        with open(ng_pkl, 'rb') as fh:
            pack['nodes_gdf'] = pickle.load(fh)
    elif os.path.exists(ng_json):
        try:
            pack['nodes_gdf'] = gpd.read_file(ng_json)
        except Exception:
            with open(ng_json, 'r', encoding='utf8') as fh:
                gj = json.load(fh)
            pack['nodes_gdf'] = gpd.GeoDataFrame.from_features(gj.get('features', []))

    # adj matrices
    for name in ('adj_geo', 'adj_faction', 'adj_matrix'):
        npz = os.path.join(graph_dir, f'{name}.npz')
        npy = os.path.join(graph_dir, f'{name}.npy')
        if os.path.exists(npz):
            try:
                import scipy.sparse as sp
                pack[name] = sp.load_npz(npz).toarray()
            except Exception:
                pack[name] = np.load(npz, allow_pickle=True)
        elif os.path.exists(npy):
            pack[name] = np.load(npy, allow_pickle=True)

    # node_features
    nf = os.path.join(graph_dir, 'node_features.npy')
    if os.path.exists(nf):
        pack['node_features'] = np.load(nf, allow_pickle=True)

    # dates
    djson = os.path.join(graph_dir, 'dates.json')
    dpkl = os.path.join(graph_dir, 'dates.pkl')
    if os.path.exists(djson):
        try:
            with open(djson, 'r', encoding='utf8') as fh:
                pack['dates'] = json.load(fh)
        except Exception:
            pass
    elif os.path.exists(dpkl):
        with open(dpkl, 'rb') as fh:
            pack['dates'] = pickle.load(fh)

    return pack


def main():
    print('Rebuilding processed_graph_data.pkl from chunks...')
    pack = load_chunks(GRAPH_DIR)
    if not pack:
        print('No chunks found or failed to load any content.')
        return 1
    # Ensure output dir exists
    out_dir = os.path.dirname(OUT_FILE)
    os.makedirs(out_dir, exist_ok=True)

    with open(OUT_FILE, 'wb') as fh:
        pickle.dump(pack, fh)

    print('Wrote', OUT_FILE)
    try:
        print('Size (bytes):', os.path.getsize(OUT_FILE))
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

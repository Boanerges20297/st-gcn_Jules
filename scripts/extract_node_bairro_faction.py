"""Extract bairro and faction for each node and save reconciled mapping.

Usage:
    python scripts/extract_node_bairro_faction.py

Outputs:
    - data/processed/node_bairro_faction_mapping.csv
    - outputs/nodes_with_bairro_faction.geojson

The script attempts several fallbacks to locate node data and polygon sources.
"""
import os
import sys
import pickle
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'node_bairro_faction_mapping.csv')
OUTPUT_GEOJSON = os.path.join(BASE_DIR, 'outputs', 'nodes_with_bairro_faction.geojson')

BAIRROS_CANDIDATES = [
    os.path.join(BASE_DIR, 'outputs', 'fortaleza_bairros_fence.geojson'),
    os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
]

INTEL_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia')


def load_nodes():
    # Prefer processed pickle
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'rb') as fh:
                obj = pickle.load(fh)
            if isinstance(obj, gpd.GeoDataFrame):
                gdf = obj
            elif isinstance(obj, pd.DataFrame):
                gdf = gpd.GeoDataFrame(obj, geometry='geometry' if 'geometry' in obj.columns else None)
            else:
                raise ValueError('Unsupported data type in processed file')
            return gdf
        except Exception as e:
            print('Warning: failed to load processed_graph_data.pkl:', e)
    # Fallbacks: try outputs geojson
    for p in [os.path.join(BASE_DIR, 'outputs', 'nodes.geojson'), os.path.join(BASE_DIR, 'outputs', 'fortaleza_bairros_fence.geojson')]:
        if os.path.exists(p):
            try:
                g = gpd.read_file(p)
                return g
            except Exception:
                continue
    raise FileNotFoundError('Could not find nodes GeoDataFrame. Ensure processed_graph_data.pkl or outputs nodes file exists.')


def ensure_crs_wgs(gdf):
    if gdf is None:
        return gdf
    try:
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf.to_crs(epsg=4326, inplace=True)
    except Exception:
        pass
    return gdf


def find_bairros_polygons():
    for p in BAIRROS_CANDIDATES:
        if os.path.exists(p):
            try:
                b = gpd.read_file(p)
                return b
            except Exception:
                continue
    return None


def load_intel_polygons():
    if not os.path.exists(INTEL_DIR):
        return None
    gdfs = []
    for fn in os.listdir(INTEL_DIR):
        if not fn.lower().endswith('.geojson'):
            continue
        p = os.path.join(INTEL_DIR, fn)
        try:
            g = gpd.read_file(p)
            if g is None or g.empty:
                continue
            # determine a sensible column for faction name
            name_col = None
            for cand in ['faction', 'fac', 'faccao', 'facção', 'Name', 'name', 'NAME', 'NOME', 'Nome']:
                if cand in g.columns:
                    name_col = cand
                    break
            if name_col is None:
                # fallback to first non-geometry column
                name_col = next((c for c in g.columns if c != 'geometry'), None)
            g = g.rename(columns={name_col: 'faction_name'}) if name_col else g
            if 'faction_name' not in g.columns:
                g['faction_name'] = os.path.splitext(fn)[0]
            gdfs.append(g[['geometry', 'faction_name']])
        except Exception:
            continue
    if not gdfs:
        return None
    allg = pd.concat(gdfs, ignore_index=True)
    allg = gpd.GeoDataFrame(allg, geometry='geometry')
    return allg


def main():
    print('Loading nodes...')
    nodes = load_nodes()
    if nodes is None or nodes.empty:
        print('No nodes loaded; aborting.')
        return
    # ensure geometry is Point centroids
    if nodes.geometry.is_empty.any():
        nodes = nodes[~nodes.geometry.is_empty]
    if nodes.geom_type.isin(['Polygon','MultiPolygon']).any():
        nodes['geometry'] = nodes.geometry.centroid
    nodes = ensure_crs_wgs(nodes)

    # Extract bairro from node attributes if present
    bairro_cols = ['bairro', 'Bairro', 'BAIRRO', 'nome', 'NOME', 'name', 'NAME', 'bairro_name']
    found = None
    for c in bairro_cols:
        if c in nodes.columns:
            found = c
            nodes['bairro_node'] = nodes[c].astype(str)
            break
    if found:
        print(f'Using node column "{found}" as bairro.')
    else:
        # spatial join with bairros polygons
        print('No bairro column found on nodes; attempting spatial join with bairros polygons...')
        bairros = find_bairros_polygons()
        if bairros is None:
            print('No bairros polygons found; bairro_node will be set to N/A')
            nodes['bairro_node'] = 'N/A'
        else:
            bairros = ensure_crs_wgs(bairros)
            # pick name column
            namecol = next((c for c in ['bairro', 'Bairro', 'BAIRRO', 'name', 'NOME', 'NOME_BAIRRO'] if c in bairros.columns), None)
            if namecol:
                bairros = bairros.rename(columns={namecol: 'bairro_name'})
            else:
                bairros['bairro_name'] = bairros.index.astype(str)
            try:
                # ensure same CRS
                nodes = nodes.to_crs(bairros.crs)
                joined = gpd.sjoin(nodes, bairros[['geometry', 'bairro_name']], how='left', predicate='within')
                joined_idx = joined.reset_index().groupby('index').first()
                nodes['bairro_node'] = joined_idx['bairro_name'].reindex(nodes.index).fillna('N/A')
            except Exception as e:
                print('Spatial join failed:', e)
                nodes['bairro_node'] = 'N/A'

    # Load intelligence polygons and spatially assign faction
    print('Loading intelligence polygons...')
    intel = load_intel_polygons()
    if intel is None:
        print('No intelligence polygons found; faction_spatial set to N/A')
        nodes['faction_spatial'] = 'N/A'
    else:
        intel = ensure_crs_wgs(intel)
        try:
            nodes = nodes.to_crs(intel.crs)
            joined = gpd.sjoin(nodes, intel[['geometry', 'faction_name']], how='left', predicate='within')
            # sjoin may expand rows; take first match per node index
            joined_idx = joined.reset_index().groupby('index').first()
            nodes['faction_spatial'] = joined_idx['faction_name'].reindex(nodes.index).fillna('N/A')
        except Exception as e:
            print('Intelligence spatial join failed:', e)
            nodes['faction_spatial'] = 'N/A'

    # Reconcile with existing 'faction' column if present
    if 'faction' in nodes.columns:
        nodes['faction_existing'] = nodes['faction'].astype(str)
    else:
        nodes['faction_existing'] = None

    def choose_final(r):
        s = r.get('faction_spatial')
        e = r.get('faction_existing')
        if s and s != 'N/A' and str(s).strip() != 'nan':
            return s
        if e and str(e).strip().upper() not in ['', 'NONE', 'N/A', 'NAN', 'NATURAL']:
            return e
        return 'N/A'

    nodes['faction_final'] = nodes.apply(choose_final, axis=1)

    # Summary
    total = len(nodes)
    by_faction = nodes['faction_final'].value_counts(dropna=False).to_dict()
    print(f'Total nodes: {total}')
    print('Faction counts (final):')
    for k, v in by_faction.items():
        print(f'  {k}: {v}')

    # Save CSV
    out_df = nodes.drop(columns=[c for c in nodes.columns if c == 'geometry']).copy()
    try:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        out_df.to_csv(OUTPUT_CSV, index=True)
        print('Saved CSV to', OUTPUT_CSV)
    except Exception as e:
        print('Failed to save CSV:', e)

    # Save geojson with selected columns
    try:
        os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
        out_g = nodes.copy()
        out_g = out_g.to_crs(epsg=4326)
        out_g.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
        print('Saved GeoJSON to', OUTPUT_GEOJSON)
    except Exception as e:
        print('Failed to save GeoJSON:', e)


if __name__ == '__main__':
    main()

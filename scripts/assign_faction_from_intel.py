"""Simple script: extract bairro names from intel geojsons and assign faction field to nodes.

For each intelligence geojson:
  - Extract neighborhood names from feature "name" property
  - Match to official nodes
  - Assign faction = geojson filename (without extension)

Output:
  - outputs/nodes_with_faction_assigned.geojson
"""
import os
import re
import json
import unicodedata
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
INTEL_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia')
OUT_GEOJSON = os.path.join(BASE_DIR, 'outputs', 'nodes_with_faction_assigned.geojson')


def normalize(s):
    """Normalize string for comparison."""
    if s is None:
        return ''
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def load_official_nodes():
    """Load nodes from bairros_centros_latlong.json as GeoDataFrame."""
    if not os.path.exists(NODES_FILE):
        raise FileNotFoundError(f'Nodes file not found: {NODES_FILE}')
    
    with open(NODES_FILE, 'r', encoding='utf-8') as fh:
        raw = json.load(fh)
    
    records = []
    for name, info in raw.items():
        lat = info.get('lat') or info.get('latitude')
        lon = info.get('long') or info.get('lon') or info.get('longitude')
        records.append({
            'bairro': name,
            'bairro_norm': normalize(name),
            'faction': 'N/A',
            'geometry': Point(lon, lat)
        })
    
    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs='EPSG:4326')
    return gdf


def extract_bairros_from_intel_file(filepath, faction_name):
    """Load geojson and extract (bairro_norm, faction) pairs."""
    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        print(f'  Warning: failed to read {os.path.basename(filepath)}: {e}')
        return {}
    
    bairro_faction = {}  # bairro_norm -> faction_name
    for idx, row in gdf.iterrows():
        # Try to get 'name' property
        name = None
        if 'properties' in row and isinstance(row['properties'], dict):
            name = row['properties'].get('name') or row['properties'].get('Name')
        else:
            # Try as column
            for col in ['name', 'Name', 'NAME']:
                if col in gdf.columns:
                    name = row[col]
                    break
        
        if name:
            name_norm = normalize(name)
            if name_norm:
                bairro_faction[name_norm] = faction_name
    
    return bairro_faction


def main():
    print('Loading official nodes...')
    nodes = load_official_nodes()
    print(f'  Loaded {len(nodes)} nodes')
    
    # Build mapping: bairro_norm -> faction (from all intel files)
    all_faction_map = {}
    
    if not os.path.exists(INTEL_DIR):
        print(f'No inteligencia directory at {INTEL_DIR}')
    else:
        intel_files = sorted([f for f in os.listdir(INTEL_DIR) if f.lower().endswith('.geojson')])
        print(f'Processing {len(intel_files)} intelligence geojsons...')
        
        for fn in intel_files:
            filepath = os.path.join(INTEL_DIR, fn)
            faction_name = os.path.splitext(fn)[0]  # Remove .geojson
            print(f'  {fn}...')
            
            bairro_map = extract_bairros_from_intel_file(filepath, faction_name)
            print(f'    Found {len(bairro_map)} bairro(s)')
            
            # Merge into all_faction_map (later files can overwrite earlier ones if same bairro)
            all_faction_map.update(bairro_map)
    
    # Assign faction to nodes
    print('\nAssigning factions to nodes...')
    assigned = 0
    for idx, row in nodes.iterrows():
        bn = row['bairro_norm']
        if bn in all_faction_map:
            nodes.at[idx, 'faction'] = all_faction_map[bn]
            assigned += 1
    
    print(f'  Assigned faction to {assigned} / {len(nodes)} nodes')
    
    # Show summary
    faction_counts = nodes['faction'].value_counts().to_dict()
    print('\nFaction summary:')
    for faction, count in sorted(faction_counts.items(), key=lambda x: -x[1]):
        print(f'  {faction}: {count}')
    
    # Save output
    os.makedirs(os.path.dirname(OUT_GEOJSON), exist_ok=True)
    nodes.to_file(OUT_GEOJSON, driver='GeoJSON')
    print(f'\nSaved output to: {OUT_GEOJSON}')


if __name__ == '__main__':
    main()

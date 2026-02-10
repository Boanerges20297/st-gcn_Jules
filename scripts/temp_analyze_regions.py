import json

print('='*60)
print('ANÁLISE DOS TOP20 MICRO-NÓS ATUAIS')
print('='*60)

with open('outputs/top20_micro_nodes.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

por_regiao = {'capital': [], 'rmf': [], 'interior': []}
for feat in data['features']:
    region = feat['properties'].get('region', 'unknown')
    por_regiao[region].append(feat)

for region, feats in por_regiao.items():
    print(f'\n{region.upper()}: {len(feats)} features')
    for i, f in enumerate(feats[:5]):
        props = f['properties']
        coords = f['geometry']['coordinates']
        name = props['name'][:40].ljust(40)
        print(f'  {props["rank"]}. {name} (source: {props["source"]}, lon={coords[0]:.4f}, lat={coords[1]:.4f})')
    if len(feats) > 5:
        print(f'  ... +{len(feats)-5} mais')

# Agora analisa os dados brutos de inteligencia
print('\n' + '='*60)
print('ANÁLISE DOS DADOS BRUTOS (todas as facções)')
print('='*60)

from pathlib import Path
from math import radians, sin, cos, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

INT_DIR = Path('data/raw/inteligencia')

for fp in INT_DIR.glob('*.geojson'):
    print(f'\n{fp.name}:')
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    for feat in data.get('features', []):
        if feat['geometry'].get('type') == 'Polygon' or feat['geometry'].get('type') == 'MultiPolygon':
            count += 1
    
    print(f'  Total de features: {len(data.get("features", []))}')
    print(f'  Polygons: {count}')
    
    # Análise de distribuição geográfica
    if data.get('features'):
        print(f'  Primeiras 3 features:')
        for feat in data['features'][:3]:
            props = feat.get('properties', {})
            name = props.get('nome') or props.get('name') or 'Unknown'
            geom = feat.get('geometry', {})
            
            # Calcula centroide simples
            if geom.get('type') == 'Polygon':
                coords = geom['coordinates'][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                center_lon = sum(lons) / len(lons)
                center_lat = sum(lats) / len(lats)
            elif geom.get('type') == 'MultiPolygon':
                all_coords = []
                for poly in geom['coordinates']:
                    all_coords.extend(poly[0])
                lons = [c[0] for c in all_coords]
                lats = [c[1] for c in all_coords]
                center_lon = sum(lons) / len(lons)
                center_lat = sum(lats) / len(lats)
            else:
                continue
            
            d_fortaleza = haversine(center_lon, center_lat, -38.5267, -3.8100)
            print(f'    - {name[:40]:40} (lon={center_lon:.4f}, lat={center_lat:.4f}, dist_fortaleza={d_fortaleza/1000:.1f}km)')

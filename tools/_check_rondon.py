import json
from math import radians, sin, cos, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371000 * 2 * asin(sqrt(a))

FORTALEZA_CENTER = (-38.5267, -3.8100)

d = json.load(open('data/raw/inteligencia/micronodos_faccoes_2026.geojson', encoding='utf-8'))
features = d.get('features', [])
rondon = [f for f in features if 'MARECHAL' in str(f.get('properties', {}).get('area_oficial','')).upper()]
for f in rondon[:5]:
    coords = f['geometry']['coordinates']
    gt = f['geometry']['type']
    if gt == 'Point':
        lon, lat = coords[0], coords[1]
    elif gt == 'Polygon':
        ring = coords[0]
        lon = sum(p[0] for p in ring) / len(ring)
        lat = sum(p[1] for p in ring) / len(ring)
    else:
        lon, lat = 0, 0
    dist = haversine(lon, lat, FORTALEZA_CENTER[0], FORTALEZA_CENTER[1])
    print(f"{f['properties']['micronodo']}: ({lat:.4f},{lon:.4f}) dist={dist:.0f}m")

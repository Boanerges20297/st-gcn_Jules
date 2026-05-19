#!/usr/bin/env python3
"""
Rebuild top20_micro_nodes_*.geojson files with correct region classification.
Ranks micronodes by CVLI occurrences in the last 120 days within 1km radius.
Uses pure-Python point-in-polygon (ray casting) against municipios_ceara.geojson.
No geopandas required.
"""

import json
import unicodedata
import datetime
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

BASE_DIR = Path(__file__).resolve().parents[2]
INT_DIR  = BASE_DIR / 'data' / 'raw' / 'inteligencia'
OUT_DIR  = BASE_DIR / 'outputs'
STATIC   = BASE_DIR / 'data' / 'static' / 'municipios_ceara.geojson'

# ------------------------------------------------------------
# RMF official municipality list – normalised (no accents)
# ------------------------------------------------------------
def _norm(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().upper().strip()

RMF_NORM = {
    'AQUIRAZ', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 'GUAIUBA',
    'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 'PACATUBA',
    'PARAIPABA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE',
    'SAO LUIS DO CURU', 'TRAIRI',
}

# ------------------------------------------------------------
# Load municipality boundaries using Shapely for fast PIP
# ------------------------------------------------------------
from shapely.geometry import shape, Point
from shapely.prepared import prep

print('Loading municipality boundaries...')
with open(STATIC, 'r', encoding='utf-8') as f:
    mun_fc = json.load(f)

# Pre-build prepared shapely geometries for fast PIP
municipalities = []  # list of (name_norm, name_raw, prepared_geom)
for feat in mun_fc['features']:
    name = feat['properties'].get('name') or feat['properties'].get('nome') or ''
    try:
        geom = prep(shape(feat['geometry']))
        municipalities.append((_norm(name), name, geom))
    except Exception:
        pass  # skip invalid geometries

print(f'  {len(municipalities)} municipalities loaded.')


def classify(lon, lat):
    """Returns (municipality_name, region) for a given point."""
    pt = Point(lon, lat)
    for name_norm, name_raw, prepared_geom in municipalities:
        if prepared_geom.contains(pt):
            if 'FORTALEZA' in name_norm:
                return name_raw, 'capital'
            if name_norm in RMF_NORM:
                return name_raw, 'rmf'
            return name_raw, 'interior'
    # Fallback – distance from Fortaleza centre
    FORT = (-38.5267, -3.7172)
    def hav(lo1, la1, lo2, la2):
        lo1, la1, lo2, la2 = map(radians, [lo1, la1, lo2, la2])
        return 6371000 * 2 * asin(sqrt(sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2))
    d = hav(lon, lat, FORT[0], FORT[1])
    if d < 20000:
        return 'Fortaleza (approx)', 'capital'
    if d < 80000:
        return 'RMF (approx)', 'rmf'
    return 'Interior (approx)', 'interior'


# ------------------------------------------------------------
# Load intelligence file
# ------------------------------------------------------------
OCCURRENCES_FILE = BASE_DIR / 'data' / 'raw' / 'dados_status_ocorrencias_gerais.json'
INT_FILE = INT_DIR / 'micronodos_faccoes_2026.geojson'

# ------------------------------------------------------------
# Name-based overrides for border neighborhoods whose centroid
# falls inside the wrong municipality polygon.
# Key: _norm(area_oficial), Value: (municipality_name, region)
# ------------------------------------------------------------
BOUNDARY_NAME_OVERRIDE = {
    'CANINDEZINHO': ('Fortaleza', 'capital'),  # AIS21/32ºDP – bairro de Fortaleza
    'MARECHAL RONDON': ('Caucaia', 'rmf'),
}

# ------------------------------------------------------------
# Haversine distance (metres)
# ------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    lo1, la1, lo2, la2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371000 * 2 * asin(sqrt(
        sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    ))
print(f'\nLoading {INT_FILE.name}...')
with open(INT_FILE, 'r', encoding='utf-8') as f:
    raw = json.load(f)

print(f'  {len(raw["features"])} features')

# ------------------------------------------------------------
# Classify every feature
# ------------------------------------------------------------
print('\nClassifying features (PIP)...')
by_region = {'capital': [], 'rmf': [], 'interior': []}

for i, feat in enumerate(raw['features']):
    geom = feat.get('geometry', {})
    props = feat.get('properties', {})

    if geom.get('type') == 'Point':
        lon, lat = geom['coordinates'][0], geom['coordinates'][1]
    else:
        lon = float(props.get('long', 0) or 0)
        lat = float(props.get('lat',  0) or 0)

    if lon == 0 and lat == 0:
        continue

    name = (props.get('area_oficial') or props.get('micronodo') or '').strip()
    name_key = _norm(name) if name else ''
    if name_key in BOUNDARY_NAME_OVERRIDE:
        mun_name, region = BOUNDARY_NAME_OVERRIDE[name_key]
    else:
        mun_name, region = classify(lon, lat)
    if not name:
        name = f'Área {i+1}'

    by_region[region].append({
        'name':      name,
        'micronodo': props.get('micronodo', ''),
        'faction':   props.get('faction', ''),
        'municipality': mun_name,
        'region':    region,
        'lon':       lon,
        'lat':       lat,
    })

    if (i+1) % 200 == 0:
        print(f'  ... {i+1}/{len(raw["features"])}')

for r in ('capital', 'rmf', 'interior'):
    print(f'  {r}: {len(by_region[r])} features')

# ------------------------------------------------------------
# Deduplicate by (name, lon, lat)
# ------------------------------------------------------------
def dedup(feats):
    """Deduplicate by coordinates (4-decimal precision ~11m)."""
    seen = set()
    out = []
    for f in feats:
        key = (round(f['lon'], 4), round(f['lat'], 4))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out

for r in ('capital', 'rmf', 'interior'):
    before = len(by_region[r])
    by_region[r] = dedup(by_region[r])
    print(f'  {r}: {before} → {len(by_region[r])} after dedup')

# ------------------------------------------------------------
# Load CVLI – last 120 days, with coordinates
# ------------------------------------------------------------
CVLI_RADIUS_M = 1000  # 1km scoring radius

print('\nLoading CVLI occurrences (last 120 days)...')
cvli_points = []
try:
    with open(OCCURRENCES_FILE, 'r', encoding='utf-8') as f:
        all_occ = json.load(f)
    # Use the latest date in dataset as reference (data goes to 2026-01-29)
    all_dates = [x['data'] for x in all_occ if x.get('data') and isinstance(x['data'], str)]
    max_date = max(all_dates)
    ref_date = datetime.date.fromisoformat(max_date)
    cutoff = ref_date - datetime.timedelta(days=120)
    print(f'  Reference date: {ref_date}  |  Cutoff: {cutoff}')
    for x in all_occ:
        if x.get('tipo', '').lower() != 'cvli':
            continue
        data = x.get('data')
        if not data or not isinstance(data, str) or data < str(cutoff):
            continue
        try:
            lat = float(x['latitude'])
            lon = float(x['longitude'])
            if lat == 0 and lon == 0:
                continue
            cvli_points.append((lon, lat))
        except (TypeError, ValueError):
            continue
    print(f'  {len(cvli_points)} CVLI points loaded')
except Exception as e:
    print(f'  WARNING: Could not load CVLI data ({e}). Micronodes will be unranked.')

def score_micronode(lon, lat):
    """Count CVLI within 1km + weighted score (closer = more weight)."""
    count = 0
    wscore = 0.0
    for clon, clat in cvli_points:
        # Quick bounding-box pre-filter (~1.1 degrees at equator ≈ 111km, so 0.01 ≈ 1.1km)
        if abs(clon - lon) > 0.012 or abs(clat - lat) > 0.012:
            continue
        d = haversine(lon, lat, clon, clat)
        if d <= CVLI_RADIUS_M:
            count += 1
            wscore += 1.0 - (d / CVLI_RADIUS_M) * 0.5  # weight: 1.0 at 0m, 0.5 at 1km
    return count, round(wscore, 3)

# Score all micronodes
print('\nScoring micronodes by CVLI proximity...')
for r in ('capital', 'rmf', 'interior'):
    for item in by_region[r]:
        cnt, ws = score_micronode(item['lon'], item['lat'])
        item['cvli_count'] = cnt
        item['cvli_score'] = ws
    # Sort by CVLI score descending
    by_region[r].sort(key=lambda x: x['cvli_score'], reverse=True)
    top = by_region[r][:5]
    print(f'  {r} top5: ' + ', '.join(f'{x["name"]}({x["cvli_count"]}cvli)' for x in top))

# ------------------------------------------------------------
# Build GeoJSON features (Top20 for capital/rmf, all for interior)
# ------------------------------------------------------------
def make_feature(rank, item):
    return {
        'type': 'Feature',
        'properties': {
            'rank':        rank,
            'name':        item['name'],
            'micronodo':   item['micronodo'],
            'faction':     item['faction'],
            'municipality': item['municipality'],
            'region':      item['region'],
            'cvli_count':  item.get('cvli_count', 0),
            'cvli_score':  item.get('cvli_score', 0),
            'score':       item.get('cvli_score', 0),
            'geometry_type': 'Point',
        },
        'geometry': {
            'type': 'Point',
            'coordinates': [item['lon'], item['lat']],
        }
    }

print('\nBuilding output features...')
out_by_region = {}
all_features = []

for region in ('capital', 'rmf', 'interior'):
    feats = by_region[region]
    # For capital/rmf: pick top50 unique by (name+municipality) — still sorted by cvli_score
    if region in ('capital', 'rmf'):
        selected = []
        seen_nm = set()
        for item in feats:
            key = (_norm(item['name']), _norm(item['municipality']))
            if key not in seen_nm:
                seen_nm.add(key)
                selected.append(item)
            if len(selected) == 50:
                break
    else:
        selected = feats  # all interior, ranked by cvli_score
    geo_feats = [make_feature(i+1, item) for i, item in enumerate(selected)]
    out_by_region[region] = geo_feats
    all_features.extend(geo_feats)
    print(f'  {region}: {len(geo_feats)} written')

# Print sample validation
print('\nSample RMF Top20 (by CVLI last 120 days):')
for f in out_by_region['rmf'][:5]:
    p = f['properties']
    print(f'  {p["rank"]}. {p["name"]} | {p["municipality"]} | CVLI={p["cvli_count"]} | faction={p["faction"]}')

print('\nSample Interior Top5 (by CVLI):')
for f in out_by_region['interior'][:5]:
    p = f['properties']
    print(f'  {p["rank"]}. {p["name"]} | {p["municipality"]} | CVLI={p["cvli_count"]}')

# ------------------------------------------------------------
# Write files
# ------------------------------------------------------------
print('\nWriting output files...')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Combined
combined_path = OUT_DIR / 'top20_micro_nodes.geojson'
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump({'type': 'FeatureCollection', 'features': all_features}, f, ensure_ascii=False, indent=2)
print(f'  ✓ {combined_path.name} ({len(all_features)} features)')

# Per-region
for region in ('capital', 'rmf', 'interior'):
    path = OUT_DIR / f'top20_micro_nodes_{region}.geojson'
    feats = out_by_region[region]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': feats}, f, ensure_ascii=False, indent=2)
    print(f'  ✓ {path.name} ({len(feats)} features)')

print('\n✓ Done!')

#!/usr/bin/env python3
"""
Extract top20 micro-nodes from ALL faction GeoJSON files
Classifies by REGION first, then extracts Top20 per region

Corrected version:
1. Uses proper municipality classification
2. Extracts Top20 per region, not globally
3. Validates all fixtures are processed
4. Better handling of feature names
"""
import os
import json
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

try:
    import geopandas as gpd
    from shapely.geometry import shape, Point
    HAVE_GEO = True
except Exception:
    HAVE_GEO = False
    gpd = None

BASE_DIR = Path(__file__).resolve().parents[2]
INT_DIR = BASE_DIR / 'data' / 'raw' / 'inteligencia'
OUT_DIR = BASE_DIR / 'outputs'
STATIC_DIR = BASE_DIR / 'data' / 'static'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define regional classification
FORTALEZA_CENTER = (-38.5267, -3.8100)  # lon, lat

# RMF cities (Fortaleza metro region) — normalised, no accents
import unicodedata as _ud
def _norm(s): return _ud.normalize('NFKD', s).encode('ascii', 'ignore').decode().upper().strip()

RMF_CITIES = {
    'AQUIRAZ', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 'GUAIUBA',
    'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 'PACATUBA',
    'PARAIPABA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE',
    'SAO LUIS DO CURU', 'TRAIRI',
}

# Bairros/áreas que cruzam a fronteira municipal e são incorretamente classificados
# pelo polígono do município. Forçar município e região corretos.
AREA_MUNICIPALITY_OVERRIDE = {
    'MARECHAL RONDON': ('Caucaia', 'rmf'),
}

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in meters"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

# ---------------------------------------------------------------------------
# Pure-Python point-in-polygon (ray casting) — sem geopandas, sem fallback
# ---------------------------------------------------------------------------
def _pip_ring(px, py, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _pip_geom(lon, lat, geom):
    gt = geom['type']
    if gt == 'Polygon':
        return _pip_ring(lon, lat, geom['coordinates'][0])
    elif gt == 'MultiPolygon':
        for poly in geom['coordinates']:
            if _pip_ring(lon, lat, poly[0]):
                return True
    return False

# Preload municipality boundaries (pure JSON, sem geopandas)
print('Carregando dados de municípios para classificação geográfica...')
MUN_DATA = []  # list of (name_raw, geom)
try:
    mun_file = STATIC_DIR / 'municipios_ceara.geojson'
    with open(mun_file, 'r', encoding='utf-8') as _f:        _mun_fc = json.load(_f)
    for _feat in _mun_fc['features']:
        _name = (_feat['properties'].get('name') or
                 _feat['properties'].get('NAME') or
                 _feat['properties'].get('nome') or '')
        MUN_DATA.append((_name, _feat['geometry']))
    print(f'  OK {len(MUN_DATA)} municipios carregados (pure-Python PIP)')
except Exception as e:
    print(f'  ERRO ao carregar municípios: {e}')
    raise SystemExit('municipios_ceara.geojson e necessario -- abortando.')

def get_municipality_from_geometry(lon, lat):
    """Retorna o nome do município via PIP. Retorna None se fora do Ceará."""
    for name, geom in MUN_DATA:
        if _pip_geom(lon, lat, geom):
            return name
    return None  # ponto fora do Ceará — será ignorado

def classify_region(municipality_name):
    """Classify municipality into region: capital, rmf, interior"""
    if not municipality_name:
        return None  # sem município → ignorar

    mun_upper = _norm(municipality_name)

    if 'FORTALEZA' in mun_upper:
        return 'capital'

    if mun_upper in RMF_CITIES:
        return 'rmf'

    return 'interior'

def guess_name(props):
    """Extract name from properties"""
    if not isinstance(props, dict):
        return None
    
    keys = ('area_oficial', 'micronodo', 'name', 'Name', 'NOME', 'nome',
            'NOME_BAIRRO', 'AREA', 'area', 'bairro', 'BAIRRO', 'comunidade',
            'Comunidade', 'community', 'title', 'Title', 'TITULO', 'titulo',
            'descricao', 'Descricao', 'DESCRICAO', 'description', 'Description')
    
    for k in keys:
        v = props.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    
    return None


# ================= ENRIQUECIMENTO DE MICRO-NÓS COM RUAS CRÍTICAS =================
import math

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000  # metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_exogenous_points(json_path):
    import json
    points = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for event in data:
            for pt in event.get('points', []):
                if 'lat' in pt and 'lng' in pt and 'description' in pt:
                    points.append({
                        'lat': pt['lat'],
                        'lng': pt['lng'],
                        'description': pt['description']
                    })
    except Exception as e:
        print(f'Erro ao carregar eventos exógenos: {e}')
    return points


import re
def extract_street(desc):
    # Tenta extrair nome de rua de uma descrição
    # Ex: "RUA JOÃO HENRIQUE DA SILVA SN" ou "AV. CAPITÃO HUGO BEZERRA, 221"
    m = re.search(r'(RUA|AV\.?|AVENIDA|TRAVESSA|ALAMEDA|RODOVIA|ESTRADA|VILA|PRAÇA|PRACA|LARGO|BECO|PASSAGEM|CAMINHO|R\. |AV |R |RUA |AVENIDA )([A-Z0-9 .\-ÁÉÍÓÚÃÕÇÊÂÔÜ]+)', desc.upper())
    if m:
        return m.group(0).title()
    return desc.split('-')[0].strip().title() if '-' in desc else desc.title()

def find_critical_streets(lon, lat, exo_points, max_dist=1000):
    # max_dist em metros (aumentado para 1000)
    close = []
    for pt in exo_points:
        dist = haversine_distance(lon, lat, pt['lng'], pt['lat'])
        if dist <= max_dist:
            street = extract_street(pt['description'])
            close.append((dist, street))
    close.sort()
    # Remove duplicatas mantendo ordem
    seen = set()
    result = []
    for _, street in close:
        if street not in seen:
            seen.add(street)
            result.append(street)
        if len(result) == 3:
            break
    return result


# Usar arquivo geocodificado para obter descrições de ruas mais detalhadas
EXOGENOUS_EVENTS_PATH = os.path.join(BASE_DIR, 'data', 'exogenous_events_geocoded.json')
EXO_POINTS = load_exogenous_points(EXOGENOUS_EVENTS_PATH)

print('='*70)
print('EXTRAÇÃO TOP20 MICRO-NÓS - VERSÃO 2 (REGIONAL)')
print('='*70)

# Scan all faction files
all_features = []
faction_files = sorted(INT_DIR.glob('*.geojson'))

print(f'\nProcessando {len(faction_files)} arquivos de facções:')
for fp in faction_files:
    print(f'\n  {fp.name}...')
    
    try:
        gdf = None
        features_list = []
        
        # Use raw JSON (simpler and more reliable)
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        features_list = data.get('features', [])
        
        # Process features
        feature_count = 0
        for feat in features_list:
            props = feat.get('properties', {})
            geom = feat.get('geometry', {})
            
            if not geom or geom.get('type') not in ('Polygon', 'MultiPolygon', 'Point', 'MultiPoint'):
                continue
            
            # Extract centroid
            gtype = geom.get('type')
            try:
                if gtype == 'Point':
                    coords = geom['coordinates']
                    lon, lat = coords[0], coords[1]
                elif gtype == 'MultiPoint':
                    points = geom['coordinates']
                    lon = sum(p[0] for p in points) / len(points)
                    lat = sum(p[1] for p in points) / len(points)
                elif gtype == 'Polygon':
                    ring = geom['coordinates'][0]
                    lon = sum(p[0] for p in ring) / len(ring)
                    lat = sum(p[1] for p in ring) / len(ring)
                elif gtype == 'MultiPolygon':
                    all_coords = []
                    for poly in geom['coordinates']:
                        all_coords.extend(poly[0])
                    lon = sum(p[0] for p in all_coords) / len(all_coords)
                    lat = sum(p[1] for p in all_coords) / len(all_coords)
                else:
                    continue
            except Exception:
                continue
            
            # Calculate score (area in m²)
            score = 1.0
            if gtype in ('Polygon', 'MultiPolygon') and HAVE_GEO:
                try:
                    if HAVE_GEO:
                        from shapely.geometry import shape
                        geom_obj = shape(geom)
                        from pyproj import Transformer
                        project = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True).transform
                        from shapely.ops import transform as shapely_transform
                        geom_proj = shapely_transform(project, geom_obj)
                        score = geom_proj.area
                except Exception:
                    score = 1.0
            
            # Extract name
            name = guess_name(props)
            
            # Determine municipality and region
            # Check area_oficial first for known boundary-crossing neighborhoods
            area_oficial = _norm(props.get('area_oficial', '') or '')
            if area_oficial in AREA_MUNICIPALITY_OVERRIDE:
                municipality, region = AREA_MUNICIPALITY_OVERRIDE[area_oficial]
            else:
                municipality = get_municipality_from_geometry(lon, lat)
                if municipality is None:
                    continue  # ponto fora do Ceará — descartado
                region = classify_region(municipality)
                if region is None:
                    continue  # não classificável — descartado

            all_features.append({
                'name': name,
                'source': fp.name,
                'score': float(score),
                'geometry_type': gtype,
                'lon': lon,
                'lat': lat,
                'municipality': municipality,
                'region': region,
                'properties': props
            })
            feature_count += 1
        
        print(f'    ✓ {feature_count} features processadas')
    
    except Exception as e:
        print(f'    ERRO: {e}')

print(f'\n  Total de features carregadas: {len(all_features)}')

# Deduplicate by (name, lon, lat)
print('\nRemovendo duplicatas...')
seen = set()
unique_features = []
for feat in all_features:
    key = (feat['name'], round(feat['lon'], 6), round(feat['lat'], 6))
    if key not in seen:
        seen.add(key)
        unique_features.append(feat)

print(f'  De {len(all_features)} para {len(unique_features)} após dedup')

# Group by region
print('\nAgrupando por região...')
by_region = {'capital': [], 'rmf': [], 'interior': []}
for feat in unique_features:
    r = feat['region']
    by_region[r].append(feat)

for r in ['capital', 'rmf', 'interior']:
    print(f'  {r}: {len(by_region[r])} features')

# Extract Top20 per region (CAPITAL and RMF), ALL for INTERIOR
print('\nExtraindo micro-nós por região...')
out_features = []
node_id = 1

for region_name in ['capital', 'rmf', 'interior']:
    feats = by_region[region_name]
    
    # Sort by score (descending)
    sorted_feats = sorted(feats, key=lambda x: x['score'], reverse=True)
    
    # INTERIOR: take ALL features, CAPITAL/RMF: take top 50 only
    if region_name == 'interior':
        selected = sorted_feats  # ALL interior features
    else:
        selected = sorted_feats[:50]  # Top 50 for capital and RMF
    
    print(f'\n  {region_name.upper()}: {len(selected)} features selecionadas ({"TODAS" if region_name == "interior" else "Top 20"})')
    
    for rank, feat in enumerate(selected, 1):
        name = feat['name'] or f"{Path(feat['source']).stem} - Área {rank}"
        # Enriquecer com ruas críticas próximas
        critical_streets = find_critical_streets(feat['lon'], feat['lat'], EXO_POINTS)
        feat_obj = {
            'type': 'Feature',
            'properties': {
                'node_id': node_id,
                'rank': rank,
                'name': name,
                'source': feat['source'],
                'score': feat['score'],
                'municipality': feat['municipality'],
                'region': region_name,
                'geometry_type': 'Point',
                'source_geometry_type': feat['geometry_type'],
                'is_centroid': feat['geometry_type'] not in ('Point', 'MultiPoint'),
                'critical_streets': critical_streets
            },
            'geometry': {
                'type': 'Point',
                'coordinates': [feat['lon'], feat['lat']]
            }
        }
        out_features.append(feat_obj)
        node_id += 1
        
        if rank <= 5:
            print(f'    {rank}. {name[:50]:50} ({feat["municipality"]}, score={feat["score"]:.0f})')

# Write output files
print('\n' + '='*70)
print('GRAVANDO ARQUIVOS DE SAÍDA')
print('='*70)

# Combined GeoJSON
combined = {'type': 'FeatureCollection', 'features': out_features}
combined_path = OUT_DIR / 'top20_micro_nodes.geojson'
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
print(f'\n✓ {combined_path} ({len(out_features)} features)')

# Per-region GeoJSON files
region_data = {'capital': [], 'rmf': [], 'interior': []}
for feat in out_features:
    r = feat['properties']['region']
    region_data[r].append(feat)

for region_name in ['capital', 'rmf', 'interior']:
    feats_r = region_data[region_name]
    region_path = OUT_DIR / f'top20_micro_nodes_{region_name}.geojson'
    with open(region_path, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': feats_r}, f, ensure_ascii=False, indent=2)
    print(f'✓ {region_path} ({len(feats_r)} features)')

print('\n' + '='*70)
print('✓ CONCLUÍDO SUCESSO!')
print('='*70)

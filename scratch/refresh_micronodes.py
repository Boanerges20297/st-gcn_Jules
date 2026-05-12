import os
import json
import pandas as pd
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configurações
BASE_DIR = Path(r"c:\Users\Boanerges\Desktop\Projetos\Report Preview")
CSV_PATH = BASE_DIR / "data" / "raw" / "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv"
INT_FILE = BASE_DIR / "data" / "raw" / "inteligencia" / "micronodos_faccoes_2026.geojson"
MUN_FILE = BASE_DIR / "data" / "static" / "municipios_ceara.geojson"
OUT_DIR  = BASE_DIR / "outputs"

CVLI_RADIUS_M = 1000  # Raio de 1km para pontuação
WINDOW_DAYS   = 120   # Janela de análise

def _norm(s):
    if not s: return ""
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().upper().strip()

def haversine(lon1, lat1, lon2, lat2):
    lo1, la1, lo2, la2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371000 * 2 * asin(sqrt(
        sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    ))

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

def point_in_polygon(lon, lat, geom):
    gt = geom['type']
    if gt == 'Polygon':
        return _pip_ring(lon, lat, geom['coordinates'][0])
    elif gt == 'MultiPolygon':
        for poly in geom['coordinates']:
            if _pip_ring(lon, lat, poly[0]):
                return True
    return False

def run_refresh():
    print("🚀 Iniciando atualização dos Top 30 Micronodos...")
    
    # 1. Carregar Municípios
    print("  Carregando malha municipal...")
    with open(MUN_FILE, 'r', encoding='utf-8') as f:
        mun_fc = json.load(f)
    municipalities = []
    rmf_list = {'AQUIRAZ', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 'GUAIUBA',
                'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 'PACATUBA',
                'PARAIPABA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE',
                'SAO LUIS DO CURU', 'TRAIRI'}
    for feat in mun_fc['features']:
        name = feat['properties'].get('name') or feat['properties'].get('nome') or ''
        municipalities.append((_norm(name), name, feat['geometry']))

    def classify(lon, lat):
        for name_norm, name_raw, geom in municipalities:
            if point_in_polygon(lon, lat, geom):
                if 'FORTALEZA' in name_norm: return name_raw, 'capital'
                if name_norm in rmf_list: return name_raw, 'rmf'
                return name_raw, 'interior'
        return "Desconhecido", "interior"

    # 2. Carregar Ocorrências Recentes (CSV Enriquecido)
    print(f"  Carregando ocorrências de {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df.dropna(subset=['data', 'latitude', 'longitude'])
    
    # Pegar os últimos 120 dias a partir da última data no dataset
    max_date = df['data'].max()
    cutoff = max_date - timedelta(days=WINDOW_DAYS)
    print(f"  Referência: {max_date.date()} | Cutoff: {cutoff.date()}")
    
    cvlis = df[(df['tipo'].str.lower() == 'cvli') & (df['data'] >= cutoff)]
    cvli_points = cvlis[['longitude', 'latitude']].values.tolist()
    print(f"  {len(cvli_points)} CVLIs encontrados na janela de {WINDOW_DAYS} dias.")

    # 3. Carregar Micronodos de Inteligência
    print(f"  Carregando {INT_FILE.name}...")
    with open(INT_FILE, 'r', encoding='utf-8') as f:
        raw_int = json.load(f)
    
    features_to_rank = []
    for i, feat in enumerate(raw_int['features']):
        geom = feat.get('geometry', {})
        props = feat.get('properties', {})
        
        if geom.get('type') == 'Point':
            lon, lat = geom['coordinates'][0], geom['coordinates'][1]
        else:
            lon = float(props.get('long') or props.get('longitude') or 0)
            lat = float(props.get('lat') or props.get('latitude') or 0)
        
        if lon == 0 or lat == 0: continue
        
        mun_name, region = classify(lon, lat)
        name = (props.get('area_oficial') or props.get('micronodo') or props.get('name') or f'Área {i+1}').strip()
        
        # Scoring
        count = 0
        wscore = 0.0
        for clon, clat in cvli_points:
            if abs(clon - lon) > 0.012 or abs(clat - lat) > 0.012: continue
            d = haversine(lon, lat, clon, clat)
            if d <= CVLI_RADIUS_M:
                count += 1
                wscore += 1.0 - (d / CVLI_RADIUS_M) * 0.5
        
        features_to_rank.append({
            'name': name,
            'faction': props.get('faction', 'N/A'),
            'municipality': mun_name,
            'region': region,
            'lon': lon,
            'lat': lat,
            'cvli_count': count,
            'cvli_score': round(wscore, 3)
        })

    # 4. Gerar Outputs
    print("  Gerando arquivos GeoJSON atualizados...")
    by_region = {'capital': [], 'rmf': [], 'interior': []}
    for item in features_to_rank:
        by_region[item['region']].append(item)
    
    all_final_features = []
    for region in ('capital', 'rmf', 'interior'):
        # Sort by score
        region_items = sorted(by_region[region], key=lambda x: x['cvli_score'], reverse=True)
        
        # Limitar Top 50 para Capital/RMF para garantir fluidez no mapa
        selected = region_items[:50] if region != 'interior' else region_items
        
        geo_feats = []
        for i, item in enumerate(selected):
            feat = {
                'type': 'Feature',
                'properties': {
                    'rank': i + 1,
                    'name': item['name'],
                    'faction': item['faction'],
                    'municipality': item['municipality'],
                    'region': item['region'],
                    'cvli_count': item['cvli_count'],
                    'score': item['cvli_score'],
                    'geometry_type': 'Point'
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [item['lon'], item['lat']]
                }
            }
            geo_feats.append(feat)
            all_final_features.append(feat)
        
        # Salvar por região
        out_path = OUT_DIR / f"top20_micro_nodes_{region}.geojson"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'type': 'FeatureCollection', 'features': geo_feats}, f, ensure_ascii=False, indent=2)
        print(f"    ✅ {out_path.name}: {len(geo_feats)} pontos.")

    # Salvar Combinado
    comb_path = OUT_DIR / "top20_micro_nodes.geojson"
    with open(comb_path, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': all_final_features}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✨ Atualização concluída! {len(all_final_features)} micronodos processados.")

if __name__ == "__main__":
    run_refresh()

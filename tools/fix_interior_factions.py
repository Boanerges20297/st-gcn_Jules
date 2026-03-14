"""
Script para corrigir dados do Interior:
1. Atualizar faction no bairros_centros_latlong.json com dados de inteligencia_faccoes.csv
2. Adicionar 17 municípios ausentes com coordenadas do GeoJSON
3. Reprocessar processed_interior.pkl
"""
import json
import os
import pickle
import unicodedata
import re
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BAIRROS_FILE       = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
BAIRROS_BAK        = BAIRROS_FILE + '.bak_interior_fix'
INTEL_FILE         = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia_faccoes.csv')
GEOJSON_FILE       = os.path.join(BASE_DIR, 'data', 'static', 'municipios_ceara.geojson')
OCORRENCIAS_FILE   = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.json')
OUT_PKL            = os.path.join(BASE_DIR, 'data', 'processed', 'processed_interior.pkl')

RMF_OFFICIAL = [
    'AQUIRAZ','BEBERIBE','CASCAVEL','CAUCAIA','CHOROZINHO','EUSEBIO',
    'GUAIUBA','HORIZONTE','ITAITINGA','MARACANAU','MARANGUAPE','PACAJUS',
    'PACATUBA','PARACURU','PINDORETAMA','SAO GONCALO DO AMARANTE',
    'SAO LUIS DO CURU','TRAIRI'
]

INTERIOR_NODES = [
    'ACARAU','ACOPIARA','ALTO SANTO','AMONTADA','ARACATI','ARATUBA','BARBALHA',
    'BATURITE','BOA VIAGEM','CAMOCIM','CANINDE','CARIRE','CEDRO','CRATEUS','CRATO',
    'GROAIRAS','IBICUITINGA','ICO','IGUATU','ITAPAJE','ITAPIPOCA','ITAPIUNA',
    'ITAREMA','JAGUARETAMA','JAGUARIBE','JIJOCA DE JERICOACOARA','JUAZEIRO DO NORTE',
    'LIMOEIRO DO NORTE','MIRAIMA','MORADA NOVA','PEDRA BRANCA','PENTECOSTE',
    'QUIXADA','QUIXERE','RUSSAS','SAO BENEDITO','SENADOR POMPEU','SOBRAL',
    'TABULEIRO DO NORTE','TAUA','TEJUCUOCA','TIANGUA','UBAJARA','VICOSA DO CEARA'
]

def normalize(s):
    s = unicodedata.normalize('NFD', str(s))
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^A-Z0-9 ]', '', s.upper().strip())

# ── 1. Carregar dados de facção do CSV ────────────────────────────────────────
logging.info("Carregando inteligencia_faccoes.csv ...")
intel_df = pd.read_csv(INTEL_FILE)
intel_df['local_norm'] = intel_df['local'].apply(normalize)
intel_df = intel_df[intel_df['regiao_sistema'].str.upper() == 'INTERIOR'].copy()

# Manter só registros com facção real
intel_df = intel_df[~intel_df['faccao_predominante'].isin(['NEUTRO', '', None])]
faction_map = dict(zip(intel_df['local_norm'], intel_df['faccao_predominante'].str.upper()))
logging.info(f"  {len(faction_map)} municípios do Interior com facção: {faction_map}")

# ── 2. Extrair coordenadas do GeoJSON para os 17 ausentes ─────────────────────
logging.info("Extraindo coordenadas do GeoJSON ...")
with open(GEOJSON_FILE, 'r', encoding='utf-8') as f:
    gj = json.load(f)

name_key = None
for feat in gj['features']:
    for k in feat['properties']:
        if 'nome' in k.lower() or 'name' in k.lower():
            name_key = k
            break
    if name_key:
        break

coords_from_geojson = {}
for feat in gj['features']:
    props = feat['properties']
    raw_name = str(props.get(name_key, ''))
    norm = normalize(raw_name)
    if norm not in INTERIOR_NODES:
        continue
    geom = feat['geometry']
    if geom['type'] == 'Point':
        lon, lat = geom['coordinates']
    elif geom['type'] == 'MultiPolygon':
        all_coords = [c for ring in geom['coordinates'][0] for c in ring]
        lon = sum(c[0] for c in all_coords) / len(all_coords)
        lat = sum(c[1] for c in all_coords) / len(all_coords)
    else:  # Polygon
        ring = geom['coordinates'][0]
        lon = sum(c[0] for c in ring) / len(ring)
        lat = sum(c[1] for c in ring) / len(ring)
    coords_from_geojson[norm] = {'lat': round(lat, 4), 'long': round(lon, 4)}

logging.info(f"  Coordenadas extraídas: {list(coords_from_geojson.keys())}")

# ── 3. Backup + atualizar bairros_centros_latlong.json ────────────────────────
logging.info("Fazendo backup e atualizando bairros JSON ...")
with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
    bairros = json.load(f)

# Criar mapeamento chave_exata a partir do upper
key_map = {k.upper().strip(): k for k in bairros.keys()}

updated = 0
added = 0

for node_upper in INTERIOR_NODES:
    faction = faction_map.get(node_upper, 'NEUTRO')

    if node_upper in key_map:
        # Nó existe: atualizar faction
        exact_key = key_map[node_upper]
        old_faction = bairros[exact_key].get('faction', 'NEUTRO')
        bairros[exact_key]['faction'] = faction
        bairros[exact_key]['regiao'] = 'interior'
        if old_faction != faction:
            logging.info(f"  ATUALIZADO {exact_key}: {old_faction} -> {faction}")
            updated += 1
    else:
        # Nó ausente: adicionar se tiver coordenadas
        if node_upper in coords_from_geojson:
            c = coords_from_geojson[node_upper]
            bairros[node_upper] = {
                'lat': c['lat'],
                'long': c['long'],
                'regiao': 'interior',
                'faction': faction
            }
            logging.info(f"  ADICIONADO {node_upper}: lat={c['lat']} long={c['long']} faction={faction}")
            added += 1
        else:
            logging.warning(f"  SEM COORDS para {node_upper} — ignorado")

# Backup antes de salvar
import shutil
shutil.copy2(BAIRROS_FILE, BAIRROS_BAK)
with open(BAIRROS_FILE, 'w', encoding='utf-8') as f:
    json.dump(bairros, f, ensure_ascii=False, indent=2)

logging.info(f"Bairros JSON atualizado: {updated} facções corrigidas, {added} nós adicionados")

# ── 4. Reprocessar processed_interior.pkl ────────────────────────────────────
logging.info("Reprocessando processed_interior.pkl ...")

def clean_name(n):
    n = unicodedata.normalize('NFD', str(n))
    n = ''.join(c for c in n if unicodedata.category(c) != 'Mn')
    n = re.sub(r'[^A-Z0-9 ]', '', n.upper().strip())
    return n

# Carregar ocorrências
with open(OCORRENCIAS_FILE, 'r', encoding='utf-8') as f:
    raw = json.load(f)

clean_records = []
for rec in raw:
    try:
        tipo_ev = str(rec.get('tipo_evento', '') or '').upper()
        tipo_raw = str(rec.get('tipo', '') or '').upper()
        tipo = 'cvli' if any(w in tipo_ev for w in ['HOMICIDIO','CVLI','MORTE VIOLENTA']) else tipo_raw.lower()
        data_str = str(rec.get('data', '') or rec.get('data_fato', '') or '')[:10]
        if len(data_str) < 10:
            continue
        data = pd.to_datetime(data_str, errors='coerce')
        if pd.isna(data):
            continue
        loc = clean_name(rec.get('municipio', '') or rec.get('bairro', '') or '')
        if not loc:
            continue

        def extract_scalar(key):
            v = rec.get(key)
            if isinstance(v, list):
                return v[0] if v else None
            return v

        clean_records.append({
            'data': data,
            'loc_clean': loc,
            'tipo': tipo,
            'tipo_evento': tipo_ev,
            'arma': str(extract_scalar('arma') or '').upper()
        })
    except:
        continue

occ_df = pd.DataFrame(clean_records).dropna(subset=['data'])
logging.info(f"  Ocorrências carregadas: {len(occ_df)}")

months = 1000 / 30.0
cvli_counts = occ_df[occ_df['tipo'] == 'cvli'].groupby('loc_clean').size()

# Carregar nós atualizados do bairros JSON
with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
    nodes_raw = json.load(f)

final_records = []
for name, info in nodes_raw.items():
    c_name = clean_name(name)
    if c_name == 'DIF':
        continue
    reg = info.get('regiao', 'interior').lower()
    if c_name in RMF_OFFICIAL:
        reg = 'rmf'
    elif reg == 'rmf':
        continue

    # Incluir apenas nós do Interior
    if reg != 'interior':
        continue

    has_f = info.get('faction', 'NEUTRO').upper() != 'NEUTRO'
    c_per_m = cvli_counts.get(c_name, 0) / months

    if has_f or c_per_m >= 1.0:
        final_records.append({
            'name': c_name,
            'lat': info['lat'],
            'long': info['long'],
            'regiao': 'interior',
            'faction': info.get('faction', 'NEUTRO').upper(),
            'tension_index': 0.0
        })

nodes_df = pd.DataFrame(final_records).drop_duplicates(subset=['name']).reset_index(drop=True)

# Ensure all string columns use object dtype, not StringDtype
for col in nodes_df.columns:
    if nodes_df[col].dtype == 'object':
        nodes_df[col] = nodes_df[col].astype(str)

nodes_gdf = gpd.GeoDataFrame(
    nodes_df,
    geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat),
    crs="EPSG:4326"
)
logging.info(f"  Nós do Interior no pkl: {len(nodes_gdf)}")
logging.info(f"  Facções: {nodes_gdf['faction'].value_counts().to_dict()}")

# Construir tensores
start_d = occ_df['data'].min()
end_d   = occ_df['data'].max()
date_range = pd.date_range(start_d, end_d)
date_map   = {d: i for i, d in enumerate(date_range)}

N = len(nodes_gdf)
features = np.zeros((N, len(date_range), 29))
node_map = {row['name']: i for i, row in nodes_gdf.iterrows()}

is_veiculo = occ_df['tipo_evento'].str.contains('ROUBO.*VEICULO|FURTO.*VEICULO|CARRO|MOTO', regex=True)
is_intel   = (
    occ_df['tipo_evento'].str.contains('LESAO.*BALA|DISPARO|TIRO|INVASAO', regex=True) |
    occ_df['arma'].str.contains('ARMA DE FOGO', regex=True)
)

for idx, row in occ_df.iterrows():
    if row['loc_clean'] in node_map:
        n_idx = node_map[row['loc_clean']]
        t_idx = date_map.get(row['data'])
        if t_idx is None:
            continue
        if row['tipo'] == 'cvli':
            features[n_idx, t_idx, 0] += 1
        if is_veiculo.loc[idx]:
            features[n_idx, t_idx, 1] += 1
        if is_intel.loc[idx]:
            features[n_idx, t_idx, 27] += 1

for d_idx, date in enumerate(date_range):
    features[:, d_idx, 28] = features[:, d_idx, 0].sum()
    features[:, d_idx, 3 + date.weekday()] = 1.0
    features[:, d_idx, 10 + date.month - 1] = 1.0
    if date.weekday() >= 5:
        features[:, d_idx, 22] = 1.0

for n in range(N):
    features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).mean().values
    features[n, :, 2]  = nodes_gdf.iloc[n]['tension_index']

dist_mat = cdist(nodes_gdf[['lat', 'long']].values, nodes_gdf[['lat', 'long']].values, 'euclidean')
adj_geo  = (dist_mat < 0.5).astype(float)  # 0.5 grau (~55km) para municípios do interior

# Ensure consistent dtypes before saving
nodes_gdf_save = nodes_gdf.copy()
for col in nodes_gdf_save.columns:
    if nodes_gdf_save[col].dtype == 'object' and col != 'geometry':
        nodes_gdf_save[col] = nodes_gdf_save[col].astype(str)

result = {
    'node_features': features,
    'adj_geo':       adj_geo,
    'adj_conflict':  np.eye(N),
    'nodes_gdf':     nodes_gdf_save,
    'dates':         date_range
}

with open(OUT_PKL, 'wb') as f:
    pickle.dump(result, f)

logging.info(f"processed_interior.pkl salvo: {N} nós | {len(date_range)} datas | features shape {features.shape}")

# ── 5. Resumo final ──────────────────────────────────────────────────────────
faction_counts = nodes_gdf['faction'].value_counts()
interior_tension_nodes = nodes_gdf[nodes_gdf['faction'] != 'NEUTRO']
logging.info("=== RESUMO ===")
logging.info(f"  Total nós Interior: {N}")
logging.info(f"  Nós com facção (zonas de tensão): {len(interior_tension_nodes)}")
for _, row in interior_tension_nodes.iterrows():
    logging.info(f"    {row['name']} ({row['faction']})")
logging.info(f"  Cobertura esperada (Cov@{N}): {len(interior_tension_nodes)}/{N} = {len(interior_tension_nodes)/N*100:.1f}%")

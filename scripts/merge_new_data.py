import pandas as pd
import numpy as np
import os
import re
import json
import unicodedata
import subprocess
import sys
from scipy.spatial import KDTree

# Configurações de Caminhos
BASE_DIR = os.getcwd()
OFFICIAL_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
BAIRROS_REF = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')

def normalize_text(text):
    if not text or pd.isna(text): return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def robust_load_any(path):
    """Carrega dados novos independente do formato (JSON ou CSV)."""
    if path.lower().endswith('.csv'):
        return pd.read_csv(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tentativa de extração robusta de JSON (para casos malformados)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
                    return pd.DataFrame(item['data'])
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data: return pd.DataFrame(data['data'])
            return pd.DataFrame([data])
    except:
        # Brute force regex para JSONs quebrados
        pattern = r'\{[^{}]*?"id":\s*?"\d+"[^{}]*?\}'
        matches = re.findall(pattern, content, re.DOTALL)
        records = []
        for m in matches:
            try: records.append(json.loads(m.strip().rstrip(',')))
            except: continue
        return pd.DataFrame(records)
    return pd.DataFrame()

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Iniciar Geocoder (Nominatim)
geolocator = Nominatim(user_agent="report_preview_v2")
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)
GEO_CACHE = {}

def get_street_from_coords(lat, lon):
    """Busca o nome da rua via OpenStreetMap com cache local."""
    key = f"{round(lat, 4)}_{round(lon, 4)}"
    if key in GEO_CACHE: return GEO_CACHE[key]
    
    try:
        location = geolocator.reverse((lat, lon), timeout=3)
        if location:
            address = location.raw.get('address', {})
            street = address.get('road') or address.get('suburb') or address.get('pedestrian')
            GEO_CACHE[key] = street
            return street
    except: pass
    return None

def merge(new_data_path):
    print(f"--- INICIANDO MESCLAGEM EM CSV: {new_data_path} ---")
    
    # ... (carregamento oficial e dicionario geo ja existentes) ...
    if os.path.exists(OFFICIAL_CSV):
        df_official = pd.read_csv(OFFICIAL_CSV)
    else:
        df_official = pd.DataFrame()

    df_new = robust_load_any(new_data_path)
    if df_new.empty: return

    with open(BAIRROS_REF, 'r', encoding='utf-8') as f:
        geo_ref = json.load(f)
    node_names = list(geo_ref.keys())
    node_coords = [[float(v['lat']), float(v['long'])] for v in geo_ref.values()]
    tree = KDTree(node_coords)

    # Termos de natureza para disparar geolocalizacao reversa se encontrados no campo de rua
    invalid_street_terms = ['HOMICIDIO', 'BALA', 'FOGO', 'LESAO', 'MORTE', 'CADAVER', 'LATROCINIO', 'TIRO']

    print("Enriquecendo dados (Bairro + Rua)...")
    for idx, row in df_new.iterrows():
        lat = pd.to_numeric(row.get('latitude'), errors='coerce')
        lon = pd.to_numeric(row.get('longitude'), errors='coerce')
        b_at = row.get('bairro')
        street_at = str(row.get('name', row.get('LocalOcor', ''))).upper()
        
        if not pd.isna(lat) and lat != 0:
            # 1. Enriquecimento de Bairro (KDTree)
            dist, i = tree.query([lat, lon])
            if dist < 0.05:
                df_new.at[idx, 'bairro'] = node_names[i]
            
            # 2. Enriquecimento de Rua (Geopy)
            # Dispara se rua estiver vazia ou contiver termos de natureza
            needs_reverse = (pd.isna(street_at) or len(street_at) < 4 or any(t in street_at for t in invalid_street_terms))
            if needs_reverse:
                street_found = get_street_from_coords(lat, lon)
                if street_found:
                    df_new.at[idx, 'name'] = street_found.upper()

        # Normalização
        df_new.at[idx, 'bairro'] = normalize_text(df_new.at[idx, 'bairro'])

    # ... (mesclagem e remocao de duplicatas) ...
    df_combined = pd.concat([df_official, df_new], ignore_index=True)
    df_combined['temp_key'] = df_combined['id'].astype(str) + "_" + df_combined['data'].astype(str) + "_" + df_combined['hora'].astype(str)
    df_combined = df_combined.drop_duplicates(subset=['temp_key'], keep='first').drop(columns=['temp_key'])
    
    df_combined.to_csv(OFFICIAL_CSV, index=False, encoding='utf-8')
    print(f"✅ SUCESSO! Base atualizada em {OFFICIAL_CSV}")

    # 7. Disparar processamento subsequente
    dp_path = os.path.join('src', 'core', 'data_processing.py')
    if os.path.exists(dp_path):
        print("Executando data_processing.py...")
        subprocess.run([sys.executable, dp_path], check=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/merge_new_data.py novo_arquivo.json")
    else:
        merge(sys.argv[1])

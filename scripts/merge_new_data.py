import pandas as pd
import numpy as np
import os
import re
import json
import unicodedata
import subprocess
import sys

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
import time

GEO_CACHE = {}
LAST_GEO_REQUEST = [0]

# Nominatim exige 1 segundo entre requisições (User-Agent obrigatório)
geolocator = Nominatim(user_agent="report_preview_merge_v2", timeout=10)

def get_street_from_coords(lat, lon):
    """Busca rua via Nominatim com rate limiting de 1.5seg entre requisições."""
    key = f"{round(lat, 4)}_{round(lon, 4)}"
    if key in GEO_CACHE: 
        return GEO_CACHE[key]
    
    # Rate limiting obrigatório para Nominatim (1+ segundo entre requisições)
    elapsed = time.time() - LAST_GEO_REQUEST[0]
    if elapsed < 1.5:
        time.sleep(1.5 - elapsed)
    
    try:
        LAST_GEO_REQUEST[0] = time.time()
        location = geolocator.reverse(f"{lat}, {lon}")
        if location:
            address = location.raw.get('address', {})
            # Tenta encontrar a rua em ordem de preferência
            street = address.get('road') or address.get('street') or address.get('footway') or address.get('suburb')
            if street:
                GEO_CACHE[key] = street
                return street
    except Exception as e:
        pass  # Silencioso, continue
    return None

def find_closest_bairro(lat, lon, nodes_coords, node_names):
    """Encontra o bairro mais próximo usando distância euclidiana simples (sem scipy)."""
    if not lat or not lon or pd.isna(lat) or pd.isna(lon):
        return None
    
    min_dist = float('inf')
    closest_name = None
    
    for i, (nlat, nlon) in enumerate(nodes_coords):
        dist = ((lat - nlat)**2 + (lon - nlon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = node_names[i]
            if dist < 0.05:  # Threshold de proximidade
                break
    
    return closest_name if min_dist < 0.05 else None

def build_streets_cache(df_combined):
    """Constrói cache de ruas críticas (geo_streets_cache.json) a partir do CSV mesclado."""
    cache_path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
    
    # Filtrar registros com rua, bairro e coordenadas válidas
    df_valid = df_combined[
        df_combined['name'].notna() & 
        (df_combined['name'].str.len() > 0) &
        df_combined['bairro'].notna() & 
        (df_combined['bairro'].str.len() > 0) &
        df_combined['latitude'].notna() & 
        df_combined['longitude'].notna()
    ].copy()
    
    if df_valid.empty:
        print("⚠ Nenhum registro com rua/bairro/coordenadas válidas")
        return
    
    # Normalizar campos para agrupamento
    df_valid['rua_norm'] = df_valid['name'].str.upper().str.strip()
    df_valid['bairro_norm'] = df_valid['bairro'].str.upper().str.strip()
    df_valid['latitude_num'] = pd.to_numeric(df_valid['latitude'], errors='coerce')
    df_valid['longitude_num'] = pd.to_numeric(df_valid['longitude'], errors='coerce')
    
    # Agrupar por (rua, bairro) e calcular estatísticas
    streets_list = []
    grouped = df_valid.groupby(['rua_norm', 'bairro_norm'], as_index=False).agg({
        'latitude_num': 'mean',
        'longitude_num': 'mean',
        'id': 'count'
    }).rename(columns={'id': 'ocorrencias'})
    
    for _, row in grouped.iterrows():
        if row['ocorrencias'] > 0:  # Apenas ruas com al menos 1 ocorrência
            streets_list.append({
                'rua': row['rua_norm'],
                'bairro': row['bairro_norm'],
                'cidade': 'FORTALEZA',  # Padrão
                'ocorrencias': int(row['ocorrencias']),
                'latitude': float(row['latitude_num']) if not pd.isna(row['latitude_num']) else None,
                'longitude': float(row['longitude_num']) if not pd.isna(row['longitude_num']) else None,
                'source': 'data'
            })
    
    # Ordenar por ocorrências (maior primeiro)
    streets_list.sort(key=lambda x: x['ocorrencias'], reverse=True)
    
    # Salvar cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(streets_list, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Cache de ruas críticas criado: {len(streets_list)} ruas em {cache_path}")


def merge(new_data_path):
    print(f"--- INICIANDO MESCLAGEM EM CSV: {new_data_path} ---")
    
    if os.path.exists(OFFICIAL_CSV):
        df_official = pd.read_csv(OFFICIAL_CSV, low_memory=False)
        print(f"✓ Base oficial carregada: {len(df_official)} registros")
    else:
        df_official = pd.DataFrame()
        print("⚠ Base oficial não existe, será criada nova")

    # Carregar APENAS os novos dados (dados_status.json)
    df_new = robust_load_any(new_data_path)
    if df_new.empty: 
        print("Erro: Arquivo de entrada vazio!")
        return
    
    print(f"✓ Novos dados carregados: {len(df_new)} registros")

    # Carrregar referência de bairros
    with open(BAIRROS_REF, 'r', encoding='utf-8') as f:
        geo_ref = json.load(f)
    node_names = list(geo_ref.keys())
    node_coords = [[float(v['lat']), float(v['long'])] for v in geo_ref.values()]

    # Termos de natureza para disparar geolocalizacao reversa
    invalid_street_terms = ['HOMICIDIO', 'BALA', 'FOGO', 'LESAO', 'MORTE', 'CADAVER', 'LATROCINIO', 'TIRO']

    print(f"\nEnriquecendo {len(df_new)} registros (Bairro + Rua)...")
    for idx, row in df_new.iterrows():
        lat = pd.to_numeric(row.get('latitude'), errors='coerce')
        lon = pd.to_numeric(row.get('longitude'), errors='coerce')
        b_at = row.get('bairro')
        street_at = str(row.get('name', row.get('LocalOcor', ''))).upper()
        
        if not pd.isna(lat) and not pd.isna(lon) and lat != 0 and lon != 0:
            # 1. Enriquecimento de Bairro
            bairro = find_closest_bairro(lat, lon, node_coords, node_names)
            if bairro:
                df_new.at[idx, 'bairro'] = bairro
            
            # 2. Enriquecimento de Rua (Nominatim com rate limiting)
            needs_reverse = (pd.isna(street_at) or len(street_at) < 4 or any(t in street_at for t in invalid_street_terms))
            if needs_reverse:
                street_found = get_street_from_coords(lat, lon)
                if street_found:
                    df_new.at[idx, 'name'] = street_found
                    print(f"  ✓ {idx+1}/{len(df_new)}: {street_found}")

        # Normalização
        df_new.at[idx, 'bairro'] = normalize_text(df_new.at[idx, 'bairro'])

    # Mesclagem e remoção de duplicatas
    print("\nMesclando com base oficial...")
    df_combined = pd.concat([df_official, df_new], ignore_index=True)
    df_combined['temp_key'] = df_combined['id'].astype(str) + "_" + df_combined['data'].astype(str) + "_" + df_combined['hora'].astype(str)
    df_combined = df_combined.drop_duplicates(subset=['temp_key'], keep='first').drop(columns=['temp_key'])
    
    
    df_combined.to_csv(OFFICIAL_CSV, index=False, encoding='utf-8')
    print(f"✅ SUCESSO! Base atualizada em {OFFICIAL_CSV}")

    # Construir cache de ruas críticas geolocalizadas
    print("\n📍 Construindo cache de ruas críticas...")
    build_streets_cache(df_combined)
    
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

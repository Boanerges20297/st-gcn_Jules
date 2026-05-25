import pandas as pd
import numpy as np
import os
import re
import json
import unicodedata
import subprocess
import sys

# --- AIS Lookup (Mapeamento oficial de 34 AIS) ---
try:
    sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
    from ais_lookup import AISLookup
    _AIS_LOOKUP = AISLookup(os.getcwd())
except Exception as e:
    print(f"[AIS] Aviso: AIS Lookup nao carregado: {e}")
    _AIS_LOOKUP = None
from datetime import datetime

try:
    from src.hostinger_sync import HostingerSyncManager
except ImportError:
    HostingerSyncManager = None

# Importação de Enriquecimento (V33)
sys.path.append(os.getcwd())
try:
    from src.enrichment import (
        get_day_of_week_pt, 
        is_brazil_holiday, 
        is_cvp_hot_day, 
        get_real_weather, 
        get_weather_label
    )
except ImportError:
    print("⚠️ Aviso: src.enrichment não encontrado. Algumas variáveis de clima/calendário serão ignoradas.")
    get_day_of_week_pt = lambda x: ""
    is_brazil_holiday = lambda x: False
    is_cvp_hot_day = lambda x: False
    get_real_weather = lambda x, **kw: 0.0
    get_weather_label = lambda x: "Desconhecido"

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
            export_meta = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                item_type = item.get('type')
                if item_type == 'header':
                    export_meta['type'] = item_type
                    if 'version' in item:
                        export_meta['version'] = item.get('version')
                    if 'comment' in item:
                        export_meta['comment'] = item.get('comment')
                elif item_type == 'database':
                    if 'name' in item:
                        export_meta['database'] = item.get('name')
                elif item_type == 'table':
                    if 'name' in item:
                        export_meta['table_name'] = item.get('name')
            for item in data:
                if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
                    df = pd.DataFrame(item['data'])
                    df.attrs['export_meta'] = export_meta
                    return df
            df = pd.DataFrame(data)
            df.attrs['export_meta'] = export_meta
            return df
        elif isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                df.attrs['export_meta'] = {}
                return df
            df = pd.DataFrame([data])
            df.attrs['export_meta'] = {}
            return df
    except:
        # Brute force regex para JSONs quebrados
        pattern = r'\{[^{}]*?"id":\s*?"\d+"[^{}]*?\}'
        matches = re.findall(pattern, content, re.DOTALL)
        records = []
        for m in matches:
            try: records.append(json.loads(m.strip().rstrip(',')))
            except: continue
        df = pd.DataFrame(records)
        df.attrs['export_meta'] = {}
        return df
    return pd.DataFrame()

from geopy.geocoders import Nominatim
import time

GEO_CACHE = {}
# 1. Carregar cache persistente (JSON)
cache_path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
if os.path.exists(cache_path):
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data_c = json.load(f)
            for item in data_c:
                k = f"{round(float(item['lat']), 3)}_{round(float(item['lng']), 3)}"
                GEO_CACHE[k] = item['rua']
    except: pass

# 2. MINERAR CSV OFICIAL (Enriquecimento Histórico Local)
# Com 147k registros, temos quase todas as ruas mapeadas
if os.path.exists(OFFICIAL_CSV):
    try:
        print(f"🔍 Minerando ruas de {OFFICIAL_CSV} para cache local...", flush=True)
        # Lendo apenas colunas necessárias para economizar memória
        df_hist = pd.read_csv(OFFICIAL_CSV, usecols=['latitude', 'longitude', 'name'], low_memory=False)
        df_hist = df_hist.dropna(subset=['latitude', 'longitude', 'name'])
        for _, row in df_hist.iterrows():
            k = f"{round(float(row['latitude']), 3)}_{round(float(row['longitude']), 3)}"
            if k not in GEO_CACHE:
                GEO_CACHE[k] = str(row['name']).upper()
        print(f"✅ Cache histórico reconstruído: {len(GEO_CACHE)} ruas identificadas localmente.", flush=True)
        del df_hist # Liberar memória
    except Exception as e:
        print(f"⚠️ Aviso: Falha ao minerar CSV histórico: {e}", flush=True)

LAST_GEO_REQUEST = [0]

# --- CONFIGURAÇÃO GOOGLE MAPS API ---
GOOGLE_API_KEY = "AIzaSyDiyGKvZeWK_6PYgbzOullUYAU_kGc8x6c"

def get_street_from_coords(lat, lon):
    """Busca rua via Google Maps API com alta velocidade."""
    # 1. Tenta cache local primeiro (3 casas decimais)
    key_cache = f"{round(lat, 3)}_{round(lon, 3)}"
    if key_cache in GEO_CACHE:
        return GEO_CACHE[key_cache]
    
    # 2. Consulta Google Maps (Sem rate limit de 1.5s)
    try:
        import requests
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GOOGLE_API_KEY}&language=pt-BR"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get("status") == "OK":
            results = data.get("results", [])
            if results:
                # Procurar pelo componente 'route' (rua)
                for res in results:
                    for comp in res.get("address_components", []):
                        if "route" in comp.get("types", []):
                            street = comp.get("long_name").upper()
                            GEO_CACHE[key_cache] = street
                            return street
                
                # Fallback: Usar o formatted_address se não achar 'route' específico
                full_addr = results[0].get("formatted_address", "")
                street = full_addr.split(',')[0].split('-')[0].strip().upper()
                if street:
                    GEO_CACHE[key_cache] = street
                    return street
        elif data.get("status") == "REQUEST_DENIED":
            print(f"  ❌ Erro Google API: {data.get('error_message')}", flush=True)
            return "ERRO_API_KEY"
    except Exception as e:
        print(f"  ⚠️ Erro na conexão com Google: {e}", flush=True)
    
    return None

def resolve_precise_bairro(lat, lon, polygons, nodes_coords, node_names):
    """Resolve o bairro cirurgicamente com Point-in-Polygon (PIP) e centróides de fallback."""
    if not lat or not lon or pd.isna(lat) or pd.isna(lon):
        return None
        
    # Tenta Point-in-Polygon (PIP)
    if polygons:
        try:
            from shapely.geometry import Point
            pt = Point(lon, lat)  # Point leva (longitude, latitude)
            for poly in polygons:
                if poly['geometry'].contains(pt):
                    return poly['name']
        except Exception:
            pass
            
    # Fallback para centróide mais próximo (Euclidiana)
    min_dist = float('inf')
    closest_name = None
    for i, (nlat, nlon) in enumerate(nodes_coords):
        dist = ((lat - nlat)**2 + (lon - nlon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = node_names[i]
            if dist < 0.05:
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


def incremental_update_streets_cache(df_new_rows):
    """Atualiza incrementalmente data/geo_streets_cache.json com as novas linhas enriquecidas."""
    cache_path = os.path.join(BASE_DIR, 'data', 'geo_streets_cache.json')
    # Carrega lista existente
    existing = []
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
    except Exception:
        existing = []

    # Indexar por (rua, bairro) para atualizacao rapida
    index = {}
    for item in existing:
        key = (item.get('rua'), item.get('bairro'))
        index[key] = item

    updated = False
    for _, row in df_new_rows.iterrows():
        try:
            rua = str(row.get('name', '')).upper().strip()
            bairro = str(row.get('bairro', '')).upper().strip()
            if not rua or not bairro:
                continue
            lat = None
            lon = None
            try:
                lat = float(row.get('latitude')) if not pd.isna(row.get('latitude')) else None
                lon = float(row.get('longitude')) if not pd.isna(row.get('longitude')) else None
            except Exception:
                lat = None; lon = None

            key = (rua, bairro)
            if key in index:
                item = index[key]
                # atualizar ocorrencias e media de coordenadas quando disponivel
                item['ocorrencias'] = int(item.get('ocorrencias', 1)) + 1
                if lat is not None and lon is not None:
                    # recalcula media aproximada
                    prev_lat = float(item.get('latitude')) if item.get('latitude') is not None else lat
                    prev_lon = float(item.get('longitude')) if item.get('longitude') is not None else lon
                    item['latitude'] = (prev_lat + lat) / 2
                    item['longitude'] = (prev_lon + lon) / 2
                index[key] = item
                updated = True
            else:
                new_item = {
                    'rua': rua,
                    'bairro': bairro,
                    'cidade': 'FORTALEZA',
                    'ocorrencias': 1,
                    'latitude': lat,
                    'longitude': lon,
                    'source': 'incremental'
                }
                existing.append(new_item)
                index[key] = new_item
                updated = True
        except Exception:
            continue

    if updated:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"✅ Cache de ruas incrementado: {len(existing)} entradas em {cache_path}")
        except Exception as e:
            print(f"⚠️ Falha ao salvar cache incremental: {e}")


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
    export_meta = df_new.attrs.get('export_meta', {}) if hasattr(df_new, 'attrs') else {}
    if df_new.empty: 
        print("Erro: Arquivo de entrada vazio!")
        return
    
    print(f"✓ Novos dados carregados: {len(df_new)} registros")

    with open(BAIRROS_REF, 'r', encoding='utf-8') as f:
        geo_ref = json.load(f)
    node_names = list(geo_ref.keys())
    node_coords = [[float(v['lat']), float(v['long'])] for v in geo_ref.values()]

    # Carregar referência de polígonos (nodes_polygons.geojson) para geoprocessamento preciso
    polygons = []
    polygons_path = os.path.join(BASE_DIR, 'data', 'static', 'nodes_polygons.geojson')
    if os.path.exists(polygons_path):
        try:
            from shapely.geometry import shape
            import re
            with open(polygons_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            for feat in geojson_data['features']:
                props = feat.get('properties', {})
                reg = str(props.get('region_type', '')).lower()
                if reg == 'capital':
                    raw_name = props.get('name') or props.get('Name') or ''
                    clean_name = re.split(r'\s*-\s*AIS', raw_name, flags=re.IGNORECASE)[0].strip().upper()
                    if clean_name:
                        polygons.append({
                            'name': clean_name,
                            'geometry': shape(feat['geometry'])
                        })
            print(f"✓ {len(polygons)} polígonos de bairros de Fortaleza carregados para geoprocessamento de precisão.")
        except Exception as e:
            print(f"⚠ Aviso ao carregar polígonos: {e}. Usando apenas centróides como fallback.")

    # Termos de natureza para disparar geolocalizacao reversa
    invalid_street_terms = ['HOMICIDIO', 'BALA', 'FOGO', 'LESAO', 'MORTE', 'CADAVER', 'LATROCINIO', 'TIRO']

    # --- FILTRAGEM PRÉVIA (Evitar reprocessar o que já existe) ---
    print(f"🔍 Filtrando registros inéditos...")
    if not df_official.empty and 'id' in df_official.columns:
        existing_ids = set(df_official['id'].astype(str).unique())
        df_new['id_str'] = df_new['id'].astype(str)
        df_new = df_new[~df_new['id_str'].isin(existing_ids)].copy()
        df_new = df_new.drop(columns=['id_str'])
        print(f"✨ {len(df_new)} novos registros identificados para enriquecimento.")
    
    if df_new.empty:
        print("✅ Nenhum dado novo para processar. Indo direto para convergência.")
    else:
        print(f"\nEnriquecendo {len(df_new)} registros inéditos (Bairro + Rua + Clima)...", flush=True)
        for i, (idx, row) in enumerate(df_new.iterrows()):
            if i < 5:
                print(f"  [DEBUG] Iniciando registro {i+1}...", flush=True)
            
            lat = pd.to_numeric(row.get('latitude'), errors='coerce')
            lon = pd.to_numeric(row.get('longitude'), errors='coerce')
        
            # 1. Enriquecimento Geográfico (Pular se já tiver bairro e rua válida)
            b_at = row.get('bairro')
            street_at = str(row.get('name', row.get('LocalOcor', ''))).upper()
            
            has_bairro = not pd.isna(b_at) and len(str(b_at)) > 2
            has_street = not pd.isna(street_at) and len(street_at) > 5 and not any(t in street_at for t in invalid_street_terms)

            # Lógica de Qualidade Total: Tenta Cache Histórico, senão vai para Google API
            if not has_street:
                # 1. Tenta o Mega-Cache Histórico (Instantâneo)
                key_cache = f"{round(lat, 3)}_{round(lon, 3)}"
                if key_cache in GEO_CACHE:
                    street_found = GEO_CACHE[key_cache]
                    df_new.at[idx, 'name'] = street_found
                    if (i+1) % 100 == 0 or i < 10:
                        print(f"  ⚡ {i+1}/{len(df_new)} [CACHE HISTÓRICO]: {street_found}", flush=True)
                else:
                    # 2. Rua Inédita: Chama Google Maps
                    # Incrementa contador de cota
                    merge.google_calls = getattr(merge, 'google_calls', 0) + 1
                    
                    print(f"  🌐 {i+1}/{len(df_new)} [GOOGLE #{merge.google_calls}] Consultando internet...", flush=True)
                    street_found = get_street_from_coords(lat, lon)
                    if street_found and street_found not in ["TIMEOUT_API", "FALHA"]:
                        df_new.at[idx, 'name'] = street_found
                        print(f"  ✅ {i+1}/{len(df_new)} [GOOGLE OK]: {street_found}", flush=True)
                    else:
                        print(f"  ❌ {i+1}/{len(df_new)} [GOOGLE FALHA]: Rua não encontrada.", flush=True)
            
            # Lógica de Qualidade Total para o Bairro:
            # 1. Se for Fortaleza, forçamos o cálculo baseado em lat/long se as coordenadas forem válidas
            # 2. Caso contrário, se o bairro estiver vazio, tentamos preencher
            cidade_clean = str(row.get('cidade', '')).strip().upper()
            is_fortaleza = 'FORTALEZA' in cidade_clean
            
            bairro_set = False
            if is_fortaleza and not pd.isna(lat) and not pd.isna(lon):
                bairro_resolved = resolve_precise_bairro(lat, lon, polygons, node_coords, node_names)
                if bairro_resolved:
                    df_new.at[idx, 'bairro'] = bairro_resolved
                    bairro_set = True
                    if i < 5:
                        print(f"  [DEBUG] {i+1}/{len(df_new)} [RESOLVED BAIRRO (FORTALEZA)]: {bairro_resolved}", flush=True)
            
            if not bairro_set and not has_bairro and not pd.isna(lat) and not pd.isna(lon):
                bairro_resolved = resolve_precise_bairro(lat, lon, polygons, node_coords, node_names)
                if bairro_resolved:
                    df_new.at[idx, 'bairro'] = bairro_resolved
                    bairro_set = True
                    if i < 5:
                        print(f"  [DEBUG] {i+1}/{len(df_new)} [RESOLVED BAIRRO (FALLBACK)]: {bairro_resolved}", flush=True)

            if not bairro_set and not has_bairro and (pd.isna(lat) or pd.isna(lon)):
                if (i+1) % 100 == 0 or i < 10:
                    print(f"  ⚪ {i+1}/{len(df_new)} [COORD INVÁLIDA] - Sem coordenadas para inferir bairro", flush=True)

            # Progresso resumido
            if (i+1) % 500 == 0:
                print(f"  ⏳ Progresso Geral: {i+1}/{len(df_new)} ({(i+1)/len(df_new)*100:.1f}%)")

            # Normalização de Bairro
            df_new.at[idx, 'bairro'] = normalize_text(df_new.at[idx, 'bairro'])

            # 2. Enriquecimento Temporal e Climático (V33 - Pular se já tiver clima)
            dt_str = row.get('data')
            has_clima = not pd.isna(row.get('clima'))
            
            if dt_str and not pd.isna(dt_str):
                try:
                    dt = pd.to_datetime(dt_str)
                    df_new.at[idx, 'dia_semana'] = get_day_of_week_pt(dt)
                    df_new.at[idx, 'eh_feriado'] = is_brazil_holiday(dt)
                    df_new.at[idx, 'dia_quente_cvp'] = is_cvp_hot_day(dt)
                    
                    if not has_clima:
                        c_lat = lat if not pd.isna(lat) and lat != 0 else -3.717
                        c_lon = lon if not pd.isna(lon) and lon != 0 else -38.543
                        precip = get_real_weather(dt, lat=c_lat, lon=c_lon)
                        df_new.at[idx, 'precipitacao_mm'] = float(precip) if precip is not None else 0.0
                        df_new.at[idx, 'clima'] = get_weather_label(precip)
                except Exception:
                    pass 

            # 3. Mapeamento de Novas Variáveis (Demográficas e Identificadores)
            if 'tipo_evento' in row:
                natureza_str = str(row['tipo_evento']).upper()
                df_new.at[idx, 'nature'] = natureza_str
                df_new.at[idx, 'tipo_evento'] = natureza_str
                # CRÍTICO: Mapeamento para o Canal 0 do modelo (CVLI)
                if any(term in natureza_str for term in ['CVLI', 'HOMICIDIO', 'LATROCINIO', 'FEMINICIDIO', 'LESAO CORPORAL SEGUIDA DE MORTE']):
                    df_new.at[idx, 'tipo'] = 'cvli'
            
            if 'nome_vitima' in row:
                df_new.at[idx, 'vitima'] = normalize_text(row['nome_vitima'])
                df_new.at[idx, 'nome_vitima'] = normalize_text(row['nome_vitima'])
                
            if 'sexo' in row:
                df_new.at[idx, 'sexo'] = str(row['sexo']).upper()
                
            if 'idade' in row:
                df_new.at[idx, 'idade'] = row['idade']
                
            if 'id_evento' in row:
                df_new.at[idx, 'id_evento'] = row['id_evento']
        
        # Preservar o ID original conforme solicitado
        if 'id' in row:
            df_new.at[idx, 'id'] = row['id']

    # --- AIS ENRICHMENT: Mapear AIS e Regiao RISP para novos registros ---
    if _AIS_LOOKUP is not None and not df_new.empty:
        print("[AIS] Enriquecendo coluna 'ais' nos novos registros...")
        ais_series, risp_series = _AIS_LOOKUP.resolve_series(
            df_new['cidade'] if 'cidade' in df_new.columns else pd.Series([''] * len(df_new)),
            df_new['bairro'] if 'bairro' in df_new.columns else pd.Series([''] * len(df_new))
        )
        df_new['ais'] = ais_series.values
        df_new['regiao_risp'] = risp_series.values
        matched = (df_new['ais'] != '').sum()
        print(f"  [AIS OK] {matched}/{len(df_new)} novos registros mapeados.")

    # Mesclagem incremental: apenas anexar registros inéditos ao CSV oficial
    print("\nMesclando incrementalmente com base oficial (append apenas registros novos)...")
    # construir temp_key no novo lote
    if 'id' in df_new.columns and 'data' in df_new.columns and 'hora' in df_new.columns:
        df_new['temp_key'] = df_new['id'].astype(str) + "_" + df_new['data'].astype(str) + "_" + df_new['hora'].astype(str)
    else:
        df_new['temp_key'] = df_new.index.astype(str)

    existing_temp_keys = set()
    if not df_official.empty and 'id' in df_official.columns and 'data' in df_official.columns and 'hora' in df_official.columns:
        df_official['temp_key'] = df_official['id'].astype(str) + "_" + df_official['data'].astype(str) + "_" + df_official['hora'].astype(str)
        existing_temp_keys = set(df_official['temp_key'].astype(str).unique())

    to_append = df_new[~df_new['temp_key'].astype(str).isin(existing_temp_keys)].copy()
    if to_append.empty:
        print("✅ Nenhum registro novo para anexar ao CSV oficial.")
        did_append = False
    else:
        print(f"✨ Preparando {len(to_append)} registros inéditos para anexar ao CSV oficial.")
        # Remover coluna temporária de controle para não ser gravada no CSV físico
        if 'temp_key' in to_append.columns:
            to_append = to_append.drop(columns=['temp_key'])
        if not df_official.empty and 'temp_key' in df_official.columns:
            df_official = df_official.drop(columns=['temp_key'])

        # Ajustar colunas para manter compatibilidade com CSV oficial
        if not df_official.empty:
            # Alinhar ao schema oficial sem depender de reindex(fill_value), que
            # pode falhar em algumas versoes do pandas ao criar varias colunas.
            official_columns = list(df_official.columns)
            for col in official_columns:
                if col not in to_append.columns:
                    if col in export_meta:
                        to_append[col] = export_meta[col]
                    else:
                        to_append[col] = ""
            # Sobrescrever colunas de metadados com origem do arquivo atual quando disponivel
            for meta_col in ['type', 'version', 'comment', 'database']:
                if meta_col in to_append.columns and export_meta.get(meta_col):
                    to_append[meta_col] = export_meta[meta_col]
            to_append = to_append[official_columns]

        # Se o CSV oficial existir, anexar; caso contrario, salvar novo arquivo completo
        if os.path.exists(OFFICIAL_CSV):
            to_append.to_csv(OFFICIAL_CSV, mode='a', header=False, index=False, encoding='utf-8')
            print(f"✅ {len(to_append)} registros anexados a {OFFICIAL_CSV} (modo append).")
            did_append = True
        else:
            # sem arquivo oficial previo: salvar cabeçalho completo
            to_append.to_csv(OFFICIAL_CSV, index=False, encoding='utf-8')
            print(f"✅ Arquivo oficial criado em {OFFICIAL_CSV} com {len(to_append)} registros.")
            did_append = False

    # Construir dataframe combinado em memoria para cache e validacao
    if os.path.exists(OFFICIAL_CSV):
        try:
            df_combined = pd.read_csv(OFFICIAL_CSV, low_memory=False)
        except Exception:
            df_combined = pd.concat([df_official, to_append], ignore_index=True)
    else:
        df_combined = pd.concat([df_official, to_append], ignore_index=True)

    # --- CONVERGÊNCIA DE MÚLTIPLAS MORTES (ANOMALIAS) ---
    print("Analisando e convergindo anomalias (múltiplas mortes por ocorrência)...")
    # Chave de evento robusta
    df_combined['event_key'] = (
        df_combined['data'].astype(str) + "_" + 
        df_combined['hora'].astype(str) + "_" + 
        df_combined['bairro'].fillna('').astype(str).str.upper() + "_" +
        pd.to_numeric(df_combined['latitude'], errors='coerce').fillna(0).round(3).astype(str) + "_" +
        pd.to_numeric(df_combined['longitude'], errors='coerce').fillna(0).round(3).astype(str)
    )
    
    cvli_mask = df_combined['tipo'].str.lower() == 'cvli'
    if cvli_mask.any():
        df_cvli = df_combined[cvli_mask].copy()
        df_others = df_combined[~cvli_mask].copy()
        
        # Agregar CVLIs (Convergência)
        agg_dict = {col: 'first' for col in df_cvli.columns if col not in ['event_key', 'qtd_mortes']}
        if 'id' in df_cvli.columns:
            agg_dict['id'] = lambda x: ' | '.join([str(v) for v in x if pd.notna(v)])
            
        # Privacidade (Remover nomes)
        for col_priv in ['nome_vitima', 'vitima']:
            if col_priv in agg_dict: del agg_dict[col_priv]
            if col_priv in df_others.columns: df_others = df_others.drop(columns=[col_priv])
            if col_priv in df_cvli.columns: df_cvli = df_cvli.drop(columns=[col_priv])

        df_cvli_collapsed = df_cvli.groupby('event_key').agg(agg_dict).reset_index()
        df_cvli_collapsed['qtd_mortes'] = df_cvli.groupby('event_key').size().values
        
        df_combined = pd.concat([df_others, df_cvli_collapsed], ignore_index=True)
        print(f"  ✓ {len(df_cvli) - len(df_cvli_collapsed)} registros de vítimas extras convergidos em seus eventos de origem.")
    else:
        df_combined['qtd_mortes'] = 1

    df_combined = df_combined.drop(columns=['event_key'])
    
    # --- AJUSTE DE ORDEM DE COLUNAS (Solicitação do Usuário) ---
    # Ordem: ... version, dia_semana, eh_feriado, dia_quente_cvp, clima, precipitacao_mm, ... restante
    cols = list(df_combined.columns)
    new_v33_cols = ['dia_semana', 'eh_feriado', 'dia_quente_cvp', 'clima', 'precipitacao_mm', 'qtd_mortes']
    
    # Remove as novas colunas da lista atual para reinceri-las na posição certa
    for c in new_v33_cols:
        if c in cols: cols.remove(c)
    
    if 'version' in cols:
        v_idx = cols.index('version') + 1
        # Reconstrói a lista de colunas na ordem exata
        final_cols = cols[:v_idx] + new_v33_cols + cols[v_idx:]
        df_combined = df_combined[final_cols]
    elif 'id' in cols:
        # Fallback se version não existir
        v_idx = cols.index('id') + 1
        final_cols = cols[:v_idx] + new_v33_cols + cols[v_idx:]
        df_combined = df_combined[final_cols]
    
    if not did_append:
        df_combined.to_csv(OFFICIAL_CSV, index=False, encoding='utf-8')
        print(f"✅ SUCESSO! Base atualizada em {OFFICIAL_CSV} com colunas ordenadas.")
        # Construir cache de ruas críticas geolocalizadas (reconstrução completa)
        print("\n📍 Construindo cache de ruas críticas (reconstrucao completa)...")
        build_streets_cache(df_combined)
    else:
        print("✅ CSV oficial atualizado por append; evitando reescrita completa.")
        # Atualizacao incremental do cache de ruas apenas com os novos registros
        if 'to_append' in globals() and not to_append.empty:
            print("\n📍 Atualizando cache de ruas criticamente de forma incremental...")
            incremental_update_streets_cache(to_append)
        else:
            print("\n📍 Nenhum registro novo para atualizar o cache incremental.")
    
    # 7. Disparar processamento subsequente
    dp_path = os.path.join('src', 'core', 'data_processing.py')
    if os.path.exists(dp_path):
        print("Executando data_processing.py...")
        subprocess.run([sys.executable, dp_path], check=False)

    # 8. Validação Automática com Gabarito (Novo)
    if not df_new.empty:
        try:
            perform_validation_log(df_combined, window_days=14)
        except Exception as e:
            print(f"⚠️ Erro na validação automática: {e}")

    if HostingerSyncManager is not None:
        try:
            sync_result = HostingerSyncManager(BASE_DIR).sync_data_merge_artifacts()
            if sync_result.get('status') == 'synced':
                print(f"✅ Sync Hostinger (data merge): {len(sync_result.get('uploaded_files', []))} arquivo(s) enviados")
        except Exception as e:
            print(f"⚠️ Sync Hostinger falhou após merge: {e}")

def perform_validation_log(df_eval, window_days=14):
    """
    Avalia a performance das predições contra os dados reais.
    Filtra para os últimos `window_days` dias para obter um gabarito tático real.
    Registra o resultado regional detalhado em VALIDATION_LOG.md.
    """
    print(f"\n📊 Iniciando Validação Regional Detalhada (Gabarito - Últimos {window_days} dias)...")
    
    # 1. Carregar Orquestrador para obter as predições e mapeamento regional
    try:
        from src.core.orchestrator import StateOrchestrator, normalize_name
        orchestrator = StateOrchestrator(os.getcwd())
        scores_map = orchestrator.get_combined_risk()
        if not scores_map:
            print("  - Não foi possível obter scores do Orquestrador para validação.")
            return
    except Exception as e:
        print(f"  - Erro ao carregar StateOrchestrator: {e}")
        return

    # 2. Mapeamento de Bairros para Regiões (via Orquestrador)
    node_to_region = {}
    for reg, spec in orchestrator.specialists.items():
        for _, row in spec['data']['nodes_gdf'].iterrows():
            node_to_region[normalize_name(str(row['name']))] = reg

    # 3. Preparar Ground Truth por Região
    df_eval['data'] = pd.to_datetime(df_eval['data'], errors='coerce')
    
    max_date = df_eval['data'].max()
    if pd.isna(max_date):
        print("  - Erro: Nenhuma data válida encontrada na base.")
        return
        
    cutoff_date = max_date - pd.Timedelta(days=window_days)
    mask_time = df_eval['data'] >= cutoff_date
    mask_cvli = df_eval['tipo'].str.lower() == 'cvli'
    
    cvlis = df_eval[mask_time & mask_cvli].copy()
    
    if cvlis.empty:
        print(f"  - Nenhum CVLI nos últimos {window_days} dias para validar.")
        return

    cvlis['node_norm'] = cvlis['bairro'].apply(normalize_name)
    cvlis['region'] = cvlis['node_norm'].map(node_to_region)
    
    # Registrar métricas por região
    regions = ['fortaleza', 'rmf', 'interior']
    results = []
    
    for reg in regions:
        reg_cvlis = cvlis[cvlis['region'] == reg]
        total_bruto = len(reg_cvlis)
        
        # Obter Top 10/20 da região
        reg_nodes = [n for n, r in node_to_region.items() if r == reg]
        reg_scores = {n: scores_map.get(n, 0.0) for n in reg_nodes}
        top_pred = sorted(reg_scores.keys(), key=lambda x: reg_scores[x], reverse=True)
        
        # Gabarito (bairros com crime na nova entrada)
        gt_bairros = set(reg_cvlis['node_norm'].unique())
        
        # Cálculo de Hits (P@k)
        hits10 = len(gt_bairros.intersection(set(top_pred[:10])))
        hits20 = len(gt_bairros.intersection(set(top_pred[:20])))
        
        p10 = hits10 / 10.0
        p20 = hits20 / 20.0
        r10 = hits10 / total_bruto if total_bruto > 0 else 0.0
        r20 = hits20 / total_bruto if total_bruto > 0 else 0.0
        
        results.append({
            'region': reg.upper(),
            'total': total_bruto,
            'hits': hits10,
            'p10': f"{p10*100:.1f}%",
            'p20': f"{p20*100:.1f}%",
            'r10': f"{r10*100:.1f}%",
            'r20': f"{r20*100:.1f}%",
            'status': "✅" if p10 >= 0.4 else ("⚠️" if p10 >= 0.2 else "🚨")
        })

    # 4. Registrar no VALIDATION_LOG.md
    log_path = os.path.join(os.getcwd(), 'VALIDATION_LOG.md')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    start_d = cvlis['data'].min().strftime('%Y-%m-%d')
    end_d = cvlis['data'].max().strftime('%Y-%m-%d')
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n### 🔄 Sessão de Validação: {now_str}\n")
        f.write(f"**Período Gabarito:** {start_d} a {end_d}\n\n")
        f.write("| Região    | N_CVLI Bruto | Hits Bruto | P@10  |  P@20 |  R@10  |  R@20  | Status |\n")
        f.write("|:----------|:------------:|:----------:|:-----:|:-----:|:------:|:------:|:------:|\n")
        for res in results:
            region_padded = res['region'].ljust(9)
            f.write(f"| {region_padded} | {res['total']:^12} | {res['hits']:^10} | {res['p10']:^5} | {res['p20']:^5} | {res['r10']:^6} | {res['r20']:^6} | {res['status']:^6} |\n")
        f.write("\n---\n")
        
    print(f"  ✅ Validação regional concluída e registrada em {log_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/merge_new_data.py novo_arquivo.json")
    else:
        merge(sys.argv[1])

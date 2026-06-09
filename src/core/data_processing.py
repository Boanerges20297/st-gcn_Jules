import sys
import os
sys.path.append(os.getcwd())
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from math import radians, cos, sin, asin, sqrt
import re
import unicodedata
import logging

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

# --- CONFIGURAÇÃO DE CAMINHOS ISM ---
DATA_DIR = 'data/raw'
BAIRROS_FILE = os.path.join(DATA_DIR, 'bairros_centros_latlong.json')
FACCOES_FILE = os.path.join(DATA_DIR, 'inteligencia_faccoes.csv')
# Usar a base consolidada CSV conforme novo fluxo, ou fallback pro JSON
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
if not os.path.exists(OCORRENCIAS_FILE):
    OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais_ENRIQUECIDO.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

RMF_OFFICIAL = [
    'AQUIRAZ', 'BEBERIBE', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 
    'GUAIUBA', 'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 
    'PACATUBA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE', 
    'SAO LUIS DO CURU', 'TRAIRI'
]

SUBDIVISION_TO_CITY = {
    'GUADALAJARA': 'CAUCAIA', 'GUAJERU': 'CAUCAIA', 'INDUSTRIAL': 'CAUCAIA',
    'IPARANA': 'CAUCAIA', 'MARECHAL RONDON': 'CAUCAIA', 'PARQUE ALBANO': 'CAUCAIA',
    'PARQUE SOLEDADE': 'CAUCAIA', 'ALTO ALEGRE': 'MARACANAU', 'CIDADE NOVA': 'MARACANAU',
    'URUCUTUBA': 'CAUCAIA'
}

def normalize_text(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def clean_name(n):
    n = normalize_text(n)
    merges = ['CONJUNTO CEARA', 'PRAIA DO FUTURO', 'VILA MANOEL SATIRO', 'ALTO ALEGRE', 'EDSON QUEIROZ', 'JOSE WALTER']
    for m in merges:
        if m in n: return m
    n = re.sub(r'\s+[IVXLCDM]+$', '', n)
    n = re.sub(r'\s+\d+$', '', n)
    n = n.strip()
    return SUBDIVISION_TO_CITY.get(n, n)

def update_geo_streets_cache(df):
    """
    Verifica se existem novas coordenadas de CVLI no dataframe e as mapeia para o cache de ruas.
    """
    import time
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        logging.warning("⚠️ geopy não encontrado. Pulando atualização do cache de ruas.")
        return

    cache_path = 'data/geo_streets_cache.json'
    streets_data = []
    cache_coords = {}
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                streets_data = json.load(f)
                # Indexar por coordenadas arredondadas (3 casas = ~110m)
                cache_coords = {(round(float(c['lat']), 3), round(float(c['lng']), 3)): c for c in streets_data}
        except: pass

    # Filtrar CVLIs na janela de 14 dias
    df_cvli = df[df['tipo'].str.lower() == 'cvli'].copy()
    if 'latitude' not in df_cvli.columns or 'longitude' not in df_cvli.columns:
        return
    if not df_cvli.empty and 'data' in df_cvli.columns:
        cutoff = df_cvli['data'].max() - pd.Timedelta(days=14)
        df_cvli = df_cvli[df_cvli['data'] >= cutoff]
        logging.info(f"⏱️ Cache de ruas: janela 14 dias ({cutoff.date()} a {df_cvli['data'].max().date()}) — {len(df_cvli)} CVLIs.")
        
    df_cvli['lat_r'] = pd.to_numeric(df_cvli['latitude'], errors='coerce').round(3)
    df_cvli['lng_r'] = pd.to_numeric(df_cvli['longitude'], errors='coerce').round(3)
    df_cvli = df_cvli.dropna(subset=['lat_r', 'lng_r'])
    
    unique_coords = df_cvli[['lat_r', 'lng_r']].drop_duplicates()
    
    new_found = 0
    geolocator = Nominatim(user_agent="report_preview_auto_update")
    
    logging.info(f"🌐 Verificando novas coordenadas para o cache de ruas ({len(unique_coords)} pontos únicos)...")
    
    for _, row in unique_coords.iterrows():
        lat, lng = row['lat_r'], row['lng_r']
        if (lat, lng) not in cache_coords:
            try:
                time.sleep(1) # Respeitar rate limit do OSM
                location = geolocator.reverse(f"{lat}, {lng}", exactly_one=True)
                if location and location.raw and 'address' in location.raw:
                    addr = location.raw['address']
                    rua = addr.get('road', addr.get('pedestrian', addr.get('path', addr.get('suburb', ''))))
                    bairro = addr.get('suburb', addr.get('neighbourhood', addr.get('city_district', '')))
                    cidade = addr.get('city', addr.get('town', addr.get('municipality', '')))
                    
                    if rua and len(rua) > 2:
                        entry = {
                            'rua': rua.upper(),
                            'bairro': bairro.upper() if bairro else '',
                            'cidade': cidade.upper() if cidade else '',
                            'lat': lat,
                            'lng': lng,
                            'ocorrencias': 1,
                            'source': 'auto_update'
                        }
                        streets_data.append(entry)
                        cache_coords[(lat, lng)] = entry
                        new_found += 1
                        logging.info(f"📍 Nova rua mapeada: {rua} ({bairro})")
            except Exception as e:
                logging.warning(f"Erro ao geocodificar {lat}, {lng}: {e}")
                
    if new_found > 0:
        # Recalcular ocorrências totais no cache para os pontos que temos no DF
        cluster_counts = df_cvli.groupby(['lat_r', 'lng_r']).size().to_dict()
        for (lt, lg), count in cluster_counts.items():
            if (lt, lg) in cache_coords:
                # Atualizamos para o maior valor (histórico vs atual)
                cache_coords[(lt, lg)]['ocorrencias'] = max(int(count), cache_coords[(lt, lg)].get('ocorrencias', 0))

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(streets_data, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Cache de ruas atualizado: {len(streets_data)} logradouros mapeados.")

def check_and_download_mysql_data():
    """
    Verifica se há novos dados no banco de dados MySQL e baixa apenas os novos registros.
    """
    import pymysql
    import decimal
    from dotenv import load_dotenv
    load_dotenv()

    json_path = os.path.join('data', 'raw', 'dados_status.json')
    enriched_csv_path = os.path.join('data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.csv')
    max_local_id = 0
    local_data_exists = False
    
    # 1. Obter o maior ID do CSV Enriquecido Oficial para evitar reprocessar registros antigos
    if os.path.exists(enriched_csv_path):
        try:
            df_enr = pd.read_csv(enriched_csv_path, usecols=['id'], low_memory=False)
            if not df_enr.empty and 'id' in df_enr.columns:
                max_local_id = int(pd.to_numeric(df_enr['id'], errors='coerce').max())
                local_data_exists = True
                logging.info(f"ID máximo local no ENRIQUECIDO.csv: {max_local_id}")
        except Exception as e:
            logging.warning(f"Não foi possível ler ENRIQUECIDO.csv: {e}")

    # Fallback pro dados_status.json se o CSV não estiver presente
    if not local_data_exists and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
                    data_rows = item['data']
                    if data_rows:
                        ids = []
                        for row in data_rows:
                            val = row.get('id')
                            if val is not None:
                                try:
                                    ids.append(int(float(val)))
                                except ValueError:
                                    pass
                        if ids:
                            max_local_id = max(ids)
                            local_data_exists = True
                            logging.info(f"ID máximo local no dados_status.json: {max_local_id}")
        except Exception as e:
            logging.warning(f"Não foi possível ler dados_status.json local: {e}")
            
    host = os.getenv('MYSQL_HOST', '').replace('"', '')
    port = int(os.getenv('MYSQL_PORT', '3306').replace('"', ''))
    user = os.getenv('MYSQL_USER', '').replace('"', '')
    password = os.getenv('MYSQL_PASSWORD', '').replace('"', '')
    database = os.getenv('MYSQL_DATABASE', '').replace('"', '')
    
    if not host or not user or not database:
        logging.warning("Configurações de conexão MySQL incompletas no arquivo .env. Pulando verificação do banco.")
        return
        
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MAX(id) as max_id FROM dados_status")
                res = cursor.fetchone()
                max_db_id = res['max_id'] if res and res['max_id'] is not None else 0
                logging.info(f"ID máximo no banco de dados MySQL: {max_db_id}")
                
                if max_db_id > max_local_id or not local_data_exists:
                    logging.info(f"Novos dados detectados no MySQL. Baixando registros com id > {max_local_id}...")
                    cursor.execute("SELECT * FROM dados_status WHERE id > %s ORDER BY id DESC", (max_local_id,))
                    rows = cursor.fetchall()
                    
                    formatted_rows = []
                    for row in rows:
                        formatted_row = {}
                        for k, v in row.items():
                            if v is None:
                                formatted_row[k] = None
                            elif isinstance(v, (int, float, decimal.Decimal)):
                                formatted_row[k] = str(v)
                            elif hasattr(v, 'strftime'):
                                if hasattr(v, 'hour'):
                                    import datetime
                                    if isinstance(v, datetime.timedelta):
                                        total_seconds = int(v.total_seconds())
                                        hours = total_seconds // 3600
                                        minutes = (total_seconds % 3600) // 60
                                        seconds = total_seconds % 60
                                        formatted_row[k] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                                    else:
                                        formatted_row[k] = v.strftime('%H:%M:%S')
                                else:
                                    formatted_row[k] = v.strftime('%Y-%m-%d')
                            elif isinstance(v, bytes):
                                formatted_row[k] = v.decode('utf-8', errors='ignore')
                            else:
                                formatted_row[k] = str(v)
                        formatted_rows.append(formatted_row)
                    
                    json_data = [
                        {"type": "header", "version": "5.1.3", "comment": "Export to JSON plugin for PHPMyAdmin"},
                        {"type": "database", "name": database},
                        {
                            "type": "table",
                            "name": "dados_status",
                            "database": database,
                            "data": formatted_rows
                        }
                    ]
                    
                    os.makedirs(os.path.dirname(json_path), exist_ok=True)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=4)
                    logging.info(f"Sucesso! {len(formatted_rows)} registros salvos em {json_path}")
                    return True
                else:
                    logging.info("Nenhum dado novo encontrado no MySQL.")
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"Erro ao consultar banco de dados MySQL: {e}")
    return False

def process_ism_data():
    logging.info("🚀 Iniciando Rebuild ISM com Inteligência Territorial Atualizada...")
    
    # Executa a verificação e download do banco MySQL
    try:
        if check_and_download_mysql_data():
            logging.info("Disparando scripts/merge_new_data.py para mesclar os novos dados...")
            import subprocess
            merge_script = os.path.join('scripts', 'merge_new_data.py')
            if os.path.exists(merge_script):
                # Passa o arquivo dados_status.json para mesclagem
                subprocess.run([sys.executable, merge_script, os.path.join('data', 'raw', 'dados_status.json')], check=False)
                logging.info("Merge finalizado. O script de merge disparou a execução subsequente de data_processing.py.")
                sys.exit(0)
            else:
                logging.warning("Script scripts/merge_new_data.py não encontrado.")
    except Exception as e:
        logging.error(f"Erro na extração MySQL automática: {e}")


    
    # 0. Carregar Inteligência de Facções
    faccoes_dict = {}
    if os.path.exists(FACCOES_FILE):
        df_fac = pd.read_csv(FACCOES_FILE)
        for _, row in df_fac.iterrows():
            loc = clean_name(str(row['local']))
            faccoes_dict[loc] = {
                'faction': str(row['faccao_predominante']).upper(),
                'grau': float(row.get('grau_dominio', 0.5))
            }
        logging.info(f"Inteligência carregada: {len(faccoes_dict)} territórios.")
    
    # 1.1 Carregar Dados da Tropa (Intencionalidade e Amenização)
    TROPA_FILE = 'data/raw/ocorrencias_tropa_limpo_fortaleza.csv'
    df_t = pd.DataFrame()
    if os.path.exists(TROPA_FILE):
        try:
            df_t = pd.read_csv(TROPA_FILE, low_memory=False, encoding="utf-8-sig")
            # Formula de Intencionalidade conforme ChampionChallenger/Sentinela V3
            PESO_NATUREZA = {
                'APREENSAO DE ARMA DE FOGO': 15.0, 'PORTE ILEGAL ART 14': 12.0,
                'TRAFICO DE DROGAS': 8.0, 'APREENSAO DE DROGAS': 6.0,
                'APREENSAO DE ENTORPECENTES': 6.0, 'MANDADO DE PRISAO': 4.0,
                'MANDADO EM ABERTO': 3.5, 'MANDADO DE PRISAO EM ABERTO': 3.5,
                'VEICULO ROUBADO RECUPERADO': 2.5, 'VEICULO ROUBADO LOCALIZADO': 2.0,
                'ABANDONO DE MATERIAL': 1.5, 'NAO INFORMADA': 0.5,
            }
            df_t["peso_nat"] = df_t["natureza"].map(PESO_NATUREZA).fillna(0.5)
            df_t["score_intel"] = (
                df_t["qtd_armas"]*15 + np.log1p(df_t["qtd_drogas"].fillna(0))*4
                + df_t["qtd_drogas_itens"]*2 + df_t["qtd_veiculos_apreendidos"]*3
                + df_t["peso_nat"]
            ).astype("float32")
            df_t["data"] = pd.to_datetime(df_t["data"], errors="coerce")
            df_t["loc_clean"] = df_t["bairro"].apply(clean_name)
            logging.info(f"✅ Inteligência de Tropa carregada: {len(df_t)} registros.")
        except Exception as e:
            logging.error(f"❌ Erro ao carregar dados da tropa: {e}")

    # 1. Carregar Ocorrências
    clean_records = []
    if OCORRENCIAS_FILE.endswith('.csv'):
        occ_raw = pd.read_csv(OCORRENCIAS_FILE)
        for _, row in occ_raw.iterrows():
            clean_records.append({
                'data': pd.to_datetime(str(row.get('data')), errors='coerce'),
                'tipo': str(row.get('tipo', '')).lower(),
                'bairro': row.get('bairro'),
                'cidade': row.get('cidade'),
                'loc_clean': clean_name(row.get('bairro', row.get('cidade'))),
                'tipo_evento': str(row.get('tipo_evento', '')).upper(),
                'arma': str(row.get('arma', '')).upper(),
                'latitude': row.get('latitude'),
                'longitude': row.get('longitude'),
                'qtd_mortes': float(row.get('qtd_mortes', 1)) # Nova coluna de anomalias
            })
    else:
        with open(OCORRENCIAS_FILE, 'r', encoding='utf-8') as f:
            occ_raw = json.load(f)
        for item in occ_raw:
            if not isinstance(item, dict): continue
            d_dict = item.get('data', item) if isinstance(item.get('data'), dict) else item
            
            def extract_scalar(key):
                v = d_dict.get(key)
                if isinstance(v, list) and len(v) > 0: v = v[0]
                if isinstance(v, dict): v = v.get(key, next(iter(v.values())) if v else None)
                return v

            dt_val = extract_scalar('data')
            if dt_val is None: continue
            
            try:
                clean_records.append({
                    'data': pd.to_datetime(str(dt_val), errors='coerce'),
                    'tipo': str(extract_scalar('tipo') or '').lower(),
                    'bairro': extract_scalar('bairro'),
                    'cidade': extract_scalar('municipio') or extract_scalar('cidade'),
                    'loc_clean': clean_name(extract_scalar('bairro_geo') or extract_scalar('bairro') or extract_scalar('municipio') or extract_scalar('cidade')),
                    'tipo_evento': str(extract_scalar('tipo_evento') or '').upper(),
                    'arma': str(extract_scalar('arma') or '').upper(),
                    'latitude': extract_scalar('latitude'),
                    'longitude': extract_scalar('longitude'),
                    'qtd_mortes': float(extract_scalar('qtd_mortes') or 1)
                })
            except: continue
    
    occ_df = pd.DataFrame(clean_records).dropna(subset=['data'])
    
    # --- NOVO: Atualizar Cache de Ruas Geolocalizadas ---
    update_geo_streets_cache(occ_df)

    # --- FILTRAGEM DE MORTES AO ACASO (ANOMALIAS NÃO-TÁTICAS) ---
    def is_random_death(row, fac_dict):
        if row['tipo'] != 'cvli': return False
        
        # Indicadores de Ação Violenta (Tática)
        is_firearm = 'FOGO' in str(row.get('arma', '')).upper()
        is_multi = float(row.get('qtd_mortes', 1)) > 1
        
        # Indicadores de Acaso (Outliers)
        event = str(row.get('tipo_evento', '')).upper()
        is_domestic = any(t in event for t in ['FEMINICIDIO', 'PARENTE', 'BRIGA', 'VIZINHO', 'PASSIONAL', 'CULPOSO'])
        is_cold_weapon = any(t in str(row.get('arma', '')).upper() for t in ['BRANCA', 'FACA', 'PAULADA', 'ESPANCAMENTO', 'PEDRADA'])
        
        # Inteligência de Área
        loc = row['loc_clean']
        faction = fac_dict.get(loc, {}).get('faction', 'NEUTRO').upper()
        is_neutral_area = faction == 'NEUTRO'
        
        # Regra de Exclusão (Refinada): 
        # O "Tribunal do Crime" usa meios brutais (pedradas/pauladas) em áreas dominadas.
        # Só excluímos se for área NEUTRA E (Arma Branca/Doméstico/Briga) E (Apenas 1 vítima)
        if is_neutral_area and not is_firearm and not is_multi:
            if is_domestic or is_cold_weapon:
                return True
        
        return False

    initial_cvli = len(occ_df[occ_df['tipo'] == 'cvli'])
    occ_df['acaso'] = occ_df.apply(lambda r: is_random_death(r, faccoes_dict), axis=1)
    
    # Excluir mortes ao acaso para limpar o sinal preditivo
    acaso_count = len(occ_df[occ_df['acaso'] == True])
    occ_df = occ_df[occ_df['acaso'] == False].copy()
    
    if acaso_count > 0:
        logging.info(f"🛡️ Filtro de Anomalias: {acaso_count} mortes ao acaso excluídas (Sinal tático preservado).")
    
    # 2. Calcular Estatísticas de CVLI para Seleção de Nós (BLINDAGEM TEMPORAL)
    # Para evitar "Selection Leakage", o ranking de importância dos bairros
    # deve ser baseado apenas em dados até o final de 2025.
    selection_cutoff = pd.Timestamp('2025-12-31')
    end_d = occ_df['data'].max()
    two_years_ago = selection_cutoff - pd.Timedelta(days=730)
    
    logging.info(f"🛡️ Seleção de nós baseada no período: {two_years_ago.date()} até {selection_cutoff.date()}")
    
    # Ranking para seleção de nós (usando apenas dados até o cutoff)
    cvli_ranking_recent = occ_df[
        (occ_df['tipo'] == 'cvli') & 
        (occ_df['data'] >= two_years_ago) &
        (occ_df['data'] <= selection_cutoff)
    ].groupby('loc_clean').size()
    
    # Ranking histórico total apenas para metadados (não afeta seleção)
    cvli_counts_total = occ_df[occ_df['tipo'] == 'cvli'].groupby('loc_clean').size()

    # 3. Carregar e Filtrar Nós (Malha Dinâmica)
    with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    pre_records = []
    for name, info in nodes_raw.items():
        c_name = clean_name(name)
        if c_name == 'DIF': continue
        
        reg = info.get('regiao', 'interior').lower()
        city_meta = normalize_text(info.get('cidade_ibge', ''))
        
        # Sincronização de Regionalização: Prioridade RMF para municípios oficiais
        if city_meta in RMF_OFFICIAL: 
            reg = 'rmf'
        if c_name in RMF_OFFICIAL: 
            reg = 'rmf'
        
        # Usamos o ranking recente para a decisão de seleção, mas o total para metadados
        recent_count = cvli_ranking_recent.get(c_name, 0)
        total_count = cvli_counts_total.get(c_name, 0)
        
        # Inteligência de Facções
        intel = faccoes_dict.get(c_name, {})
        faction = intel.get('faction', info.get('faction', 'NEUTRO')).upper()
        tension_index = 0.5 if faction != 'NEUTRO' else 0.0
        if faction == 'DISPUTA': tension_index = 1.0
        if 'CAUCAIA' in c_name or 'MARACANAU' in c_name: tension_index = 1.0
        
        pre_records.append({
            'name': c_name, 'lat': info['lat'], 'long': info['long'],
            'regiao': reg, 'faction': faction, 'tension_index': tension_index,
            'recent_cvli': recent_count,
            'total_cvli': total_count
        })
    
    df_pool = pd.DataFrame(pre_records).drop_duplicates(subset=['name'])
    
    # Seleção Top-K por Região baseada no ranking de 2 ANOS
    final_records = []
    
    # 1. Fortaleza: Todos os bairros com CVLI >= 1 (Solicitação do Usuário)
    f_all = df_pool[(df_pool['regiao'] == 'fortaleza') & (df_pool['recent_cvli'] >= 1)]
    final_records.extend(f_all.to_dict('records'))
    
    # 2. RMF: Todos os 18 Oficiais (Ordenados por criticidade recente)
    rmf = df_pool[df_pool['regiao'] == 'rmf'].sort_values('recent_cvli', ascending=False)
    final_records.extend(rmf.to_dict('records'))
    
    # 3. Interior: Top 50 (Dinâmico 2 anos - Expandido)
    i50 = df_pool[df_pool['regiao'] == 'interior'].sort_values('recent_cvli', ascending=False).head(50)
    final_records.extend(i50.to_dict('records'))
    
    nodes_df = pd.DataFrame(final_records).reset_index(drop=True)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat), crs="EPSG:4326")
    
    logging.info(f"📊 Malha Final: Fortaleza({len(f_all)}), RMF({len(rmf)}), Interior({len(i50)})")
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat), crs="EPSG:4326")

    # 4. Construir Tensores (Otimizado)
    # Janela Dinâmica: 01/01/2022 até a última data disponível (Maio/2026+)
    start_d = pd.Timestamp('2022-01-01')
    end_d = occ_df['data'].max()
    if pd.isna(end_d): end_d = pd.Timestamp.now().floor('D')
    
    date_range = pd.date_range(start_d, end_d)
    date_map = {d: i for i, d in enumerate(date_range)}
    
    # Vetorização de booleanos
    occ_df['is_veiculo'] = occ_df['tipo_evento'].str.contains('ROUBO.*VEICULO|FURTO.*VEICULO|CARRO|MOTO', regex=True, na=False)
    occ_df['is_intel'] = occ_df['tipo_evento'].str.contains('LESAO.*BALA|DISPARO|TIRO|INVASAO', regex=True, na=False) | occ_df['arma'].str.contains('ARMA DE FOGO', regex=True, na=False)
    
    # Criar mapeamento de indices temporais para o dataframe
    occ_df['t_idx'] = occ_df['data'].map(date_map)
    occ_df = occ_df.dropna(subset=['t_idx'])
    occ_df['t_idx'] = occ_df['t_idx'].astype(int)

    for reg in ['fortaleza', 'rmf', 'interior']:
        reg_nodes = nodes_gdf[nodes_gdf['regiao'] == reg].copy().reset_index(drop=True)
        N = len(reg_nodes)
        if N == 0: continue
        
        logging.info(f"⏳ Processando tensores para {reg.upper()} ({N} nós)...")
        features = np.zeros((N, len(date_range), 37)) # Expandido para 37 (V37 Elite)
        node_map = {row['name']: i for i, row in reg_nodes.iterrows()}
        
        # Filtrar ocorrências da região ou que pertençam a cidades desta região (colapsando para a sede)
        if reg == 'rmf':
            # Na RMF, colapsamos todos os bairros para suas respectivas cidades-sede
            reg_occ = occ_df[occ_df['cidade'].str.upper().isin(node_map)].copy()
            reg_occ['n_idx'] = reg_occ['cidade'].str.upper().map(node_map)
        else:
            reg_occ = occ_df[occ_df['loc_clean'].isin(node_map)].copy()
            if not reg_occ.empty:
                reg_occ['n_idx'] = reg_occ['loc_clean'].map(node_map)
        
        # 3.1 Mapeamento de Tropa para Tensores (Canais 23 e 25)
        if not df_t.empty:
            df_t_reg = df_t[df_t['loc_clean'].isin(reg_nodes['name'])].copy()
            if not df_t_reg.empty:
                df_t_reg['n_idx'] = df_t_reg['loc_clean'].map({name: i for i, name in enumerate(reg_nodes['name'])})
                df_t_reg['t_idx'] = df_t_reg['data'].map({d: i for i, d in enumerate(date_range)})
                df_t_reg = df_t_reg.dropna(subset=['n_idx', 't_idx'])
                
                # Preenchimento Canal 25 (Intencionalidade)
                tropa_group = df_t_reg.groupby(['n_idx', 't_idx'])['score_intel'].sum()
                for (n, t), val in tropa_group.items():
                    features[int(n), int(t), 25] = val
                    # Canal 23 (Amenização fixa em 0.20 quando há ação)
                    features[int(n), int(t), 23] = 0.20

        # Preenchimento de ocorrências
        if not reg_occ.empty:
            # Canal 0: CVLI (Total de mortes, não apenas registros)
            cvli_rows = reg_occ[reg_occ['tipo'] == 'cvli']
            if not cvli_rows.empty:
                cvli_group = cvli_rows.groupby(['n_idx', 't_idx'])['qtd_mortes'].sum()
                for (n, t), val in cvli_group.items(): 
                    features[n, t, 0] = val
                    
                    # --- CHOQUE REAL POR ANOMALIA (Canal 25) ---
                    # Se houve mais de uma morte no mesmo dia/bairro, gera pulso de conflito
                    if val > 1:
                        # Adiciona sinal de ruptura proporcional (base 1.0 + 0.5 por vítima extra)
                        shock_val = 1.0 + (val - 1) * 0.5
                        features[n, t, 25] = max(features[n, t, 25], shock_val)
            
            veic_group = reg_occ[reg_occ['is_veiculo']].groupby(['n_idx', 't_idx']).size()
            for (n, t), val in veic_group.items(): features[n, t, 1] = val * 2.5
            
            intel_group = reg_occ[reg_occ['is_intel']].groupby(['n_idx', 't_idx']).size()
            for (n, t), val in intel_group.items(): features[n, t, 27] = val * 2.0
        
        # Preenchimento de Datas, Clima e Calendário
        from src.enrichment import is_brazil_holiday, is_cvp_hot_day, CACHE_FILE
        weather_cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                weather_cache = json.load(f)

        
        for d_idx, date in enumerate(date_range):
            # Dia da Semana (3-9)
            features[:, d_idx, 3 + date.weekday()] = 1.0
            if date.weekday() == 4: features[:, d_idx, 7] = 1.5 # Reforço Sexta
            
            # Meses (10-21)
            features[:, d_idx, 10 + date.month - 1] = 1.0
            
            # Fim de Semana (22)
            if date.weekday() >= 5: features[:, d_idx, 22] = 1.0
            
            # --- NOVOS CANAIS (V33) ---
            # 29: Feriado
            if is_brazil_holiday(date): features[:, d_idx, 29] = 1.0
            
            # 30: Dias Quentes CVP (01-10, 30, 31)
            if is_cvp_hot_day(date): features[:, d_idx, 30] = 1.0
            
            # 31 & 32: Clima Real (API Open-Meteo Cache)
            precip = weather_cache.get(date.date(), 0.0)
            features[:, d_idx, 31] = float(precip)
            if precip > 5.0: features[:, d_idx, 32] = 1.0 # Chuva significativa
            
        for n in range(N):
            features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).sum().values
            features[n, :, 2] = reg_nodes.iloc[n]['tension_index']
            features[n, :, 28] = features[:, :, 0].sum(axis=0)

        # --- Matriz de Adjacência Tática (Substituindo adj_geo geográfica pura) ---
        frag_scores = np.zeros(N)
        if not df_t.empty and not df_t_reg.empty:
            frag_group = df_t_reg.groupby('n_idx')['score_intel'].sum()
            for n, val in frag_group.items():
                frag_scores[int(n)] = val
        
        max_frag = frag_scores.max() if frag_scores.max() > 0 else 1.0
        frag_norm = frag_scores / max_frag

        adj_geo = np.zeros((N, N)) # Salvamos como adj_geo para compatibilidade
        adj_conflict = np.eye(N)
        
        DIST_THRESHOLD_KM = 3.0
        RIVALRY_MULTIPLIER = 2.0
        
        lats = reg_nodes['lat'].values
        lons = reg_nodes['long'].values
        factions = reg_nodes['faction'].values
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = haversine(lons[i], lats[i], lons[j], lats[j])
                    
                    # Matriz Tática (adj_geo)
                    if dist <= DIST_THRESHOLD_KM:
                        weight = 1.0 / (dist + 0.1)
                        is_enemy = (factions[i] != factions[j]) and (factions[i] != 'NEUTRO') and (factions[j] != 'NEUTRO')
                        if is_enemy:
                            weight *= RIVALRY_MULTIPLIER
                            frag_j = frag_norm[j]
                            if frag_j > 0.1:
                                weight *= (1.0 + frag_j) # Ataque direcionado à ferida
                        adj_geo[i, j] = weight
                    
                    # Matriz de Conflito Histórico (adj_conflict)
                    if dist < 6.0 and factions[i] != 'NEUTRO' and factions[j] != 'NEUTRO' and factions[i] != factions[j]:
                        adj_conflict[i, j] = 1.0
        
        os.makedirs('data/processed', exist_ok=True)
        with open(f'data/processed/processed_{reg}.pkl', 'wb') as f:
            pickle.dump({'node_features': features, 'adj_geo': adj_geo, 'adj_conflict': adj_conflict, 'nodes_gdf': reg_nodes, 'dates': date_range}, f)
        logging.info(f"✅ {reg.upper()} Concluído (V33 Features).")

if __name__ == "__main__":
    process_ism_data()

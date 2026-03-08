import json
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
import re
import unicodedata
import logging

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

    # Filtrar CVLIs com coordenadas válidas
    df_cvli = df[df['tipo'].str.lower() == 'cvli'].copy()
    if 'latitude' not in df_cvli.columns or 'longitude' not in df_cvli.columns:
        return
        
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

def process_ism_data():
    logging.info("🚀 Iniciando Rebuild ISM com Inteligência Territorial Atualizada...")
    
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
    
    # 1. Carregar Ocorrências
    clean_records = []
    if OCORRENCIAS_FILE.endswith('.csv'):
        occ_raw = pd.read_csv(OCORRENCIAS_FILE)
        for _, row in occ_raw.iterrows():
            clean_records.append({
                'data': pd.to_datetime(str(row.get('data')), errors='coerce'),
                'tipo': str(row.get('tipo', '')).lower(),
                'loc_clean': clean_name(row.get('bairro', row.get('cidade'))),
                'tipo_evento': str(row.get('tipo_evento', '')).upper(),
                'arma': str(row.get('arma', '')).upper(),
                'latitude': row.get('latitude'),
                'longitude': row.get('longitude')
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
                    'loc_clean': clean_name(extract_scalar('bairro_geo') or extract_scalar('bairro') or extract_scalar('municipio') or extract_scalar('cidade')),
                    'tipo_evento': str(extract_scalar('tipo_evento') or '').upper(),
                    'arma': str(extract_scalar('arma') or '').upper(),
                    'latitude': extract_scalar('latitude'),
                    'longitude': extract_scalar('longitude')
                })
            except: continue
    
    occ_df = pd.DataFrame(clean_records).dropna(subset=['data'])
    
    # --- NOVO: Atualizar Cache de Ruas Geolocalizadas ---
    update_geo_streets_cache(occ_df)
    
    # 2. Calcular Estatísticas de CVLI
    start_d, end_d = occ_df['data'].min(), occ_df['data'].max()
    # Diferença exata em meses para o cálculo da taxa mensal
    total_months = (end_d.year - start_d.year) * 12 + (end_d.month - start_d.month) + 1
    logging.info(f"📅 Período analisado: {total_months} meses ({start_d.date()} a {end_d.date()})")
    
    cvli_counts = occ_df[occ_df['tipo'] == 'cvli'].groupby('loc_clean').size()

    # 3. Carregar e Filtrar Nós (Malha Expandida: F40, RMF18, I20)
    with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    pre_records = []
    for name, info in nodes_raw.items():
        c_name = clean_name(name)
        if c_name == 'DIF': continue
        
        reg = info.get('regiao', 'interior').lower()
        if c_name in RMF_OFFICIAL: reg = 'rmf'
        elif reg == 'rmf': continue
        
        count = cvli_counts.get(c_name, 0)
        
        # Inteligência de Facções
        intel = faccoes_dict.get(c_name, {})
        faction = intel.get('faction', info.get('faction', 'NEUTRO')).upper()
        tension_index = 0.5 if faction != 'NEUTRO' else 0.0
        if faction == 'DISPUTA': tension_index = 1.0
        if 'CAUCAIA' in c_name or 'MARACANAU' in c_name: tension_index = 1.0
        
        pre_records.append({
            'name': c_name, 'lat': info['lat'], 'long': info['long'],
            'regiao': reg, 'faction': faction, 'tension_index': tension_index,
            'total_cvli': count
        })
    
    df_pool = pd.DataFrame(pre_records).drop_duplicates(subset=['name'])
    
    # Seleção Top-K por Região
    final_records = []
    
    # 1. Fortaleza: Top 40
    f40 = df_pool[df_pool['regiao'] == 'fortaleza'].sort_values('total_cvli', ascending=False).head(40)
    final_records.extend(f40.to_dict('records'))
    
    # 2. RMF: Todos (18)
    rmf = df_pool[df_pool['regiao'] == 'rmf']
    final_records.extend(rmf.to_dict('records'))
    
    # 3. Interior: Top 20
    i20 = df_pool[df_pool['regiao'] == 'interior'].sort_values('total_cvli', ascending=False).head(20)
    final_records.extend(i20.to_dict('records'))
    
    nodes_df = pd.DataFrame(final_records).reset_index(drop=True)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat), crs="EPSG:4326")
    
    logging.info(f"📊 Malha Final: Fortaleza({len(f40)}), RMF({len(rmf)}), Interior({len(i20)})")
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat), crs="EPSG:4326")

    # 4. Construir Tensores (Otimizado)
    start_d, end_d = occ_df['data'].min(), occ_df['data'].max()
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
        features = np.zeros((N, len(date_range), 29))
        node_map = {row['name']: i for i, row in reg_nodes.iterrows()}
        
        # Filtrar ocorrencias da regiao
        reg_occ = occ_df[occ_df['loc_clean'].isin(node_map)].copy()
        reg_occ['n_idx'] = reg_occ['loc_clean'].map(node_map)
        
        # Preenchimento vetorizado via agrupação
        # Canal 0: CVLI
        cvli_group = reg_occ[reg_occ['tipo'] == 'cvli'].groupby(['n_idx', 't_idx']).size()
        for (n, t), val in cvli_group.items(): features[n, t, 0] = val
        
        # Canal 1: Veiculos
        veic_group = reg_occ[reg_occ['is_veiculo']].groupby(['n_idx', 't_idx']).size()
        for (n, t), val in veic_group.items(): features[n, t, 1] = val
        
        # Canal 27: Intel
        intel_group = reg_occ[reg_occ['is_intel']].groupby(['n_idx', 't_idx']).size()
        for (n, t), val in intel_group.items(): features[n, t, 27] = val
        
        # Preenchimento de Datas e Calendário (Vetorizado)
        for d_idx, date in enumerate(date_range):
            features[:, d_idx, 3 + date.weekday()] = 1.0
            features[:, d_idx, 10 + date.month - 1] = 1.0
            if date.weekday() >= 5: features[:, d_idx, 22] = 1.0
            
        for n in range(N):
            # Canal 24: Média Móvel (Tensão Padrão)
            features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).mean().values
            # Canal 2: Tensão de Facções (Estático do mapeamento)
            features[n, :, 2] = reg_nodes.iloc[n]['tension_index']
            # Canal 28: Somatório Global (Contexto)
            features[n, :, 28] = features[:, :, 0].sum(axis=0)

        # Matrizes de Adjacência
        dist_mat = cdist(reg_nodes[['lat', 'long']].values, reg_nodes[['lat', 'long']].values, 'euclidean')
        adj_geo = (dist_mat < 0.05).astype(float)
        
        adj_conflict = np.eye(N)
        # Otimização do loop de conflito
        factions = reg_nodes['faction'].values
        for i in range(N):
            if factions[i] == 'NEUTRO': continue
            # Encontrar vizinhos
            neighbors = np.where(dist_mat[i] < 0.06)[0]
            for j in neighbors:
                if i != j and factions[j] != 'NEUTRO' and factions[i] != factions[j]:
                    adj_conflict[i, j] = 1.0
        
        with open(f'data/processed/processed_{reg}.pkl', 'wb') as f:
            pickle.dump({'node_features': features, 'adj_geo': adj_geo, 'adj_conflict': adj_conflict, 'nodes_gdf': reg_nodes, 'dates': date_range}, f)
        logging.info(f"✅ {reg.upper()} Concluído.")

if __name__ == "__main__":
    process_ism_data()

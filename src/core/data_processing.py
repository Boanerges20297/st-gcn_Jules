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
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais_ENRIQUECIDO.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

RMF_OFFICIAL = [
    'AQUIRAZ', 'BEBERIBE', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 
    'GUAIUBA', 'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 
    'PACATUBA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE', 
    'SAO LUIS DO CURU', 'TRAIRI'
]

# Mapeamento de Consolidação (Feedback Jules)
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

def process_ism_data():
    logging.info("🚀 Iniciando Rebuild ISM...")
    
    # 1. Carregar Ocorrências
    with open(OCORRENCIAS_FILE, 'r', encoding='utf-8') as f:
        occ_raw = json.load(f)
    
    clean_records = []
    for item in occ_raw:
        if not isinstance(item, dict): continue
        
        # Localiza o dicionário de dados (item ou item['data'])
        d_dict = item.get('data', item) if isinstance(item.get('data'), dict) else item
        
        def extract_scalar(key):
            v = d_dict.get(key)
            # Se for lista, pega o primeiro
            if isinstance(v, list) and len(v) > 0: v = v[0]
            # Se ainda for dict (raro), tenta pegar valor interno
            if isinstance(v, dict): v = v.get(key, next(iter(v.values())) if v else None)
            return v

        dt_val = extract_scalar('data')
        if dt_val is None: continue
        
        try:
            # Garante que passamos uma string ou número simples para o to_datetime
            clean_records.append({
                'data': pd.to_datetime(str(dt_val), errors='coerce'),
                'tipo': str(extract_scalar('tipo') or '').lower(),
                'loc_clean': clean_name(extract_scalar('bairro_geo') or extract_scalar('bairro') or extract_scalar('municipio') or extract_scalar('cidade')),
                'tipo_evento': str(extract_scalar('tipo_evento') or '').upper(),
                'arma': str(extract_scalar('arma') or '').upper()
            })
        except: continue
    
    occ_df = pd.DataFrame(clean_records).dropna(subset=['data'])
    
    # 2. Calcular Estatísticas de CVLI
    months = 1000 / 30.0 # Janela padrão
    cvli_counts = occ_df[occ_df['tipo'] == 'cvli'].groupby('loc_clean').size()

    # 3. Carregar e Filtrar Nós (Lógica Idêntica à Auditoria)
    with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    final_records = []
    for name, info in nodes_raw.items():
        c_name = clean_name(name)
        if c_name == 'DIF': continue
        
        reg = info.get('regiao', 'interior').lower()
        if c_name in RMF_OFFICIAL: reg = 'rmf'
        elif reg == 'rmf': continue # Consolidação RMF
        
        has_f = info.get('faction', 'NEUTRO').upper() != 'NEUTRO'
        c_per_m = cvli_counts.get(c_name, 0) / months
        
        keep = False
        if reg == 'rmf' and c_name in RMF_OFFICIAL: keep = True
        elif has_f: keep = True
        elif reg == 'fortaleza' and c_per_m >= 1.0: keep = True
        elif reg == 'interior' and c_per_m >= 1.0: keep = True
        
        if keep:
            final_records.append({
                'name': c_name, 'lat': info['lat'], 'long': info['long'],
                'regiao': reg, 'faction': info.get('faction', 'NEUTRO').upper(),
                'tension_index': 1.0 if ('CAUCAIA' in c_name or 'MARACANAU' in c_name) else 0.0
            })
    
    nodes_df = pd.DataFrame(final_records).drop_duplicates(subset=['name']).reset_index(drop=True)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.long, nodes_df.lat), crs="EPSG:4326")

    # 4. Construir Tensores
    start_d, end_d = occ_df['data'].min(), occ_df['data'].max()
    date_range = pd.date_range(start_d, end_d)
    date_map = {d: i for i, d in enumerate(date_range)}
    
    is_veiculo = occ_df['tipo_evento'].str.contains('ROUBO.*VEICULO|FURTO.*VEICULO|CARRO|MOTO', regex=True)
    is_intel = occ_df['tipo_evento'].str.contains('LESAO.*BALA|DISPARO|TIRO|INVASAO', regex=True) | occ_df['arma'].str.contains('ARMA DE FOGO', regex=True)

    for reg in ['fortaleza', 'rmf', 'interior']:
        reg_nodes = nodes_gdf[nodes_gdf['regiao'] == reg].copy().reset_index(drop=True)
        N = len(reg_nodes)
        if N == 0: continue
        
        features = np.zeros((N, len(date_range), 29))
        node_map = {row['name']: i for i, row in reg_nodes.iterrows()}
        
        for _, row in occ_df.iterrows():
            if row['loc_clean'] in node_map:
                n_idx, t_idx = node_map[row['loc_clean']], date_map[row['data']]
                if row['tipo'] == 'cvli': features[n_idx, t_idx, 0] += 1
                if is_veiculo.loc[_]: features[n_idx, t_idx, 1] += 1
                if is_intel.loc[_]: features[n_idx, t_idx, 27] += 1
        
        for d_idx, date in enumerate(date_range):
            features[:, d_idx, 28] = features[:, d_idx, 0].sum()
            features[:, d_idx, 3 + date.weekday()] = 1.0
            features[:, d_idx, 10 + date.month - 1] = 1.0
            if date.weekday() >= 5: features[:, d_idx, 22] = 1.0
            
        for n in range(N):
            features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).mean().values
            features[n, :, 2] = reg_nodes.iloc[n]['tension_index']

        dist_mat = cdist(reg_nodes[['lat', 'long']].values, reg_nodes[['lat', 'long']].values, 'euclidean')
        adj_geo = (dist_mat < 0.05).astype(float)
        
        with open(f'data/processed/processed_{reg}.pkl', 'wb') as f:
            pickle.dump({'node_features': features, 'adj_geo': adj_geo, 'adj_conflict': np.eye(N), 'nodes_gdf': reg_nodes, 'dates': date_range}, f)
        logging.info(f"✅ {reg.upper()}: {N} nós.")

if __name__ == "__main__":
    process_ism_data()

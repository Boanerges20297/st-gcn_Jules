import json
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point
import unicodedata
from scipy.spatial.distance import cdist
import re

# Configuração de Caminhos
DATA_DIR = 'data/raw'
INTELIGENCIA_DIR = os.path.join(DATA_DIR, 'inteligencia')
BAIRROS_FILE = os.path.join(DATA_DIR, 'bairros_centros_latlong.json')
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais_ENRIQUECIDO.json')
EXOGENOUS_FILE = 'data/exogenous_events.json' 

def normalize_text(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def load_all_nodes(filepath):
    """Carrega os 299 nós oficiais (121 Fortaleza, 18 RMF, 160 Interior)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    # Lista de limpeza reduzida para permitir bairros de alta atividade
    TO_REMOVE = ['DIF III', 'PRECABURA', 'RACHEL DE QUEIROZ']
    INTERIOR_ZERO = ['BAIXIO', 'GRANJEIRO', 'IPAUMIRIM']
    
    for name, info in data.items():
        norm_name = name.upper().strip()
        if norm_name in TO_REMOVE or norm_name in INTERIOR_ZERO: continue
        
        # Merge de Subdivisões - Lógica Agressiva
        if 'CONJUNTO CEARA' in norm_name: norm_name = 'CONJUNTO CEARA'
        if 'PRAIA DO FUTURO' in norm_name: norm_name = 'PRAIA DO FUTURO'
        if 'VILA MANOEL SATIRO' in norm_name: norm_name = 'MANOEL SATIRO'
        
        # Remove sufixos romanos e numerais soltos no final
        norm_name = re.sub(r'\s+[IVXLCDM]+$', '', norm_name)
        norm_name = re.sub(r'\s+\d+$', '', norm_name)
        norm_name = norm_name.strip()
        
        records.append({
            'name': norm_name,
            'latitude': info['lat'],
            'longitude': info['long'],
            'regiao': info.get('regiao', 'interior').lower()
        })
    
    df = pd.DataFrame(records).drop_duplicates(subset=['name'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    return gdf, gdf.to_crs(epsg=3857)

def load_faction_layers(directory):
    layers = {}
    files = {'CV': 'COMANDO VERMELHO.geojson', 'TCP': 'TERCEIRO COMANDO PURO.geojson', 'GDE': 'GDE.geojson', 'PCC': 'PCC.geojson', 'MASSA': 'MASSA.geojson', 'GHOST': 'TERRITÓRIOS FANTASMAS.geojson', 'DISPUTA': 'COMUNIDADES EM DISPUTA.geojson'}
    for key, filename in files.items():
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
                layers[key] = gdf.to_crs(epsg=3857)
            except: layers[key] = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
        else: layers[key] = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    return layers

def calculate_intelligence(nodes_proj, layers):
    factions, tensions = [], []
    disputa_union = layers['DISPUTA'].union_all() if not layers['DISPUTA'].empty else None
    for idx, row in nodes_proj.iterrows():
        point, name = row.geometry, row['name']
        val_t = 0.0
        if disputa_union and point.distance(disputa_union) < 1000: val_t = 1.0
        if 'CAUCAIA' in name: val_t = 1.0
        tensions.append(val_t)
        found_f, max_intersections = 'NEUTRO', 0
        for f_name in ['CV', 'GDE', 'PCC', 'TCP', 'MASSA']:
            if not layers[f_name].empty:
                dist = 5000 if row['regiao'] != 'fortaleza' else 1000
                matches = layers[f_name][layers[f_name].distance(point) < dist]
                if len(matches) > max_intersections:
                    max_intersections = len(matches)
                    found_f = f_name
        if 'CAUCAIA' in name: found_f = 'CV'
        factions.append(found_f)
    return np.array(factions), np.array(tensions)

def build_feature_tensor(nodes_gdf, occurrences_df, start_date, end_date):
    date_range = pd.date_range(start_date, end_date)
    num_nodes, num_timesteps = len(nodes_gdf), len(date_range)
    features = np.zeros((num_nodes, num_timesteps, 29))
    name_to_idx = {normalize_text(name): i for i, name in enumerate(nodes_gdf['name'])}
    is_cvli = occurrences_df['tipo'].fillna('').astype(str).str.lower() == 'cvli'
    is_veiculo = occurrences_df['tipo_upper'].str.contains('ROUBO.*VEICULO|FURTO.*VEICULO|CARRO|MOTO', regex=True)
    
    # GATILHO DE INTELIGÊNCIA (Canal 27)
    # Considera:
    # 1. Palavras-chave de violência armada na descrição (LESÃO, TIRO, DISPARO)
    # 2. Uso explícito de ARMA DE FOGO no campo 'arma'
    has_keywords = occurrences_df['tipo_upper'].str.contains('LESAO.*BALA|ARMA.*FOGO|DISPARO|TIRO|INVASAO', regex=True)
    has_firearm = occurrences_df['arma'].fillna('').astype(str).str.upper().str.contains('ARMA DE FOGO', regex=True)
    is_intel_trigger = has_keywords | has_firearm
    
    df_v = occurrences_df[(occurrences_df['data'] >= start_date) & (occurrences_df['data'] <= end_date)].copy()
    df_v['day_idx'] = (df_v['data'] - start_date).dt.days
    
    # Helper para normalizar nome do bairro da ocorrência igual ao do nó
    def clean_occ_bairro(raw_name):
        n = normalize_text(raw_name)
        if not n: return None
        if 'CONJUNTO CEARA' in n: n = 'CONJUNTO CEARA'
        if 'PRAIA DO FUTURO' in n: n = 'PRAIA DO FUTURO'
        if 'VILA MANOEL SATIRO' in n: n = 'MANOEL SATIRO'
        n = re.sub(r'\s+[IVXLCDM]+$', '', n)
        n = re.sub(r'\s+\d+$', '', n)
        return n.strip()

    for _, row in df_v.iterrows():
        b_raw = row.get('bairro_geo') or row.get('municipio') or row.get('bairro')
        b_clean = clean_occ_bairro(b_raw)
        
        n_idx = name_to_idx.get(b_clean)
        
        if n_idx is not None:
            day = row['day_idx']
            if is_cvli.loc[row.name]: features[n_idx, day, 0] += 1
            if is_veiculo.loc[row.name]: features[n_idx, day, 1] += 1
            if is_intel_trigger.loc[row.name]: features[n_idx, day, 27] += 1
    
    for d_idx, date in enumerate(date_range):
        features[:, d_idx, 28] = features[:, d_idx, 0].sum()
        features[:, d_idx, 3 + date.weekday()] = 1.0
        features[:, d_idx, 10 + date.month - 1] = 1.0
        if date.weekday() >= 5: features[:, d_idx, 22] = 1.0
    
    for n in range(num_nodes):
        features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).mean().values
    
    features[:, :, 2] = nodes_gdf['tension_index'].values[:, np.newaxis]
    return features, date_range

def save_regional_dataset(nodes_gdf, features, adj_geo, adj_conf, dates, region_name, output_path):
    if region_name != 'global':
        mask = nodes_gdf['regiao'] == region_name
        r_nodes = nodes_gdf[mask].reset_index(drop=True)
        r_features = features[mask, :, :]
        coords = np.array(list(zip(r_nodes.geometry.x, r_nodes.geometry.y)))
        r_adj_geo = (cdist(coords, coords) <= 5000).astype(float)
        r_adj_conf = np.eye(len(r_nodes))
    else:
        r_nodes, r_features, r_adj_geo, r_adj_conf = nodes_gdf, features, adj_geo, adj_conf
    data_pack = {
        'node_features': r_features, 'adj_geo': r_adj_geo, 'adj_conflict': r_adj_conf,
        'dates': dates, 'nodes_gdf': r_nodes,
        'feature_names': ['CVLI', 'VEHICLE', 'TENSION', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'WEEKEND', 'SUPPRESSION', 'NORMAL_SHOCK', 'CRITICAL_SHOCK', 'INCURSION', 'INTEL_TRIGGER', 'CITY_PULSE']
    }
    with open(output_path, 'wb') as f: pickle.dump(data_pack, f)
    print(f"✅ Dataset {region_name.upper()} gerado: {len(r_nodes)} nos.")

def main():
    print("🚀 Iniciando REBUILD GLOBAL (299 Nos)...")
    nodes_gdf, nodes_proj = load_all_nodes(BAIRROS_FILE)
    
    print("📊 Carregando Ocorrencias Master...")
    with open(OCORRENCIAS_FILE, 'r', encoding='utf-8') as f:
        occ_data = json.load(f)
    clean_records = []
    for item in occ_data:
        if not isinstance(item, dict): continue
        if 'data' in item and isinstance(item['data'], dict): item = item['data']
        clean_item = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in item.items()}
        clean_records.append(clean_item)
    occ_df = pd.DataFrame(clean_records)
    occ_df['data'] = pd.to_datetime(occ_df['data'].astype(str), errors='coerce')
    occ_df = occ_df.dropna(subset=['data'])
    occ_df['tipo_upper'] = occ_df.get('tipo_evento', pd.Series()).fillna('').astype(str).str.upper()
    
    # Precisamos calcular a inteligência ANTES do filtro de ruído
    layers = load_faction_layers(INTELIGENCIA_DIR)
    factions, tensions = calculate_intelligence(nodes_proj, layers)
    nodes_gdf['faction'], nodes_gdf['tension_index'] = factions, tensions

    # --- NOISE FILTERING (BAIRRO LEVEL) ---
    # Filtrar bairros com base em crimes recentes
    # Threshold MODERADO: 0.5/mês (Equilíbrio entre massa de dados e ruído)
    print("🧹 Filtrando ruídos (Threshold: 0.5 CVLI/mês + Proteção de Facções)...")
    max_date = occ_df['data'].max()
    min_date = max_date - pd.Timedelta(days=1000)
    
    is_cvli_all = occ_df['tipo'].fillna('').astype(str).str.lower() == 'cvli'
    cvli_df = occ_df[is_cvli_all].copy()
    
    def clean_occ_bairro(raw_name):
        if not raw_name: return None
        n = normalize_text(raw_name)
        if 'CONJUNTO CEARA' in n: n = 'CONJUNTO CEARA'
        if 'PRAIA DO FUTURO' in n: n = 'PRAIA DO FUTURO'
        if 'VILA MANOEL SATIRO' in n: n = 'MANOEL SATIRO'
        n = re.sub(r'\s+[IVXLCDM]+$', '', n)
        n = re.sub(r'\s+\d+$', '', n)
        return n.strip()
    
    def clean_occ_bairro(row):
        # Fallback robusto para mapeamento de localidades
        b_raw = row.get('bairro_geo') or row.get('bairro') or row.get('municipio') or row.get('cidade')
        if not b_raw: return None
        n = normalize_text(str(b_raw))
        if 'CONJUNTO CEARA' in n: n = 'CONJUNTO CEARA'
        if 'PRAIA DO FUTURO' in n: n = 'PRAIA DO FUTURO'
        if 'VILA MANOEL SATIRO' in n: n = 'MANOEL SATIRO'
        n = re.sub(r'\s+[IVXLCDM]+$', '', n)
        n = re.sub(r'\s+\d+$', '', n)
        return n.strip()
    
    cvli_df['b_clean'] = cvli_df.apply(clean_occ_bairro, axis=1)
    
    # Calcular média por mês (baseado nos 1000 dias de janela do dataset)
    node_cvli_counts = cvli_df.groupby('b_clean').size()
    months = 1000 / 30.0
    
    # --- FILTRAGEM ESTRITA JULES ---
    valid_mask = []
    # Janela de análise de 1000 dias (aprox 33 meses)
    months = 1000 / 30.0
    
    print(f"🧹 Aplicando Filtro Jules (Meses na base: {months:.1f})")
    
    for _, row in nodes_gdf.iterrows():
        name_norm = normalize_text(row['name'])
        count = node_cvli_counts.get(name_norm, 0)
        has_faction = row.get('faction', 'NEUTRO') != 'NEUTRO'
        regiao = str(row['regiao']).lower()
        
        crime_per_month = count / months
        
        # REGRAS JULES:
        # 1. FACÇÃO: Sempre manter (Não Neutro)
        if has_faction:
            valid_mask.append(True)
        # 2. RMF: Sempre manter todos os 18 nós
        elif regiao == 'rmf':
            valid_mask.append(True)
        # 3. FORTALEZA: Crimes >= 2.0/mês
        elif regiao == 'fortaleza' and crime_per_month >= 2.0:
            valid_mask.append(True)
        # 4. INTERIOR: Crimes >= 1.0/mês
        elif regiao == 'interior' and crime_per_month >= 1.0:
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    
    nodes_gdf = nodes_gdf[valid_mask].reset_index(drop=True)
    nodes_proj = nodes_proj[valid_mask].reset_index(drop=True)
    
    # Auditoria por Região
    counts_by_reg = nodes_gdf['regiao'].value_counts()
    print(f"📉 Nós Finais Mantidos: {len(nodes_gdf)} total")
    for reg, val in counts_by_reg.items():
        print(f"   - {reg.upper()}: {val} nos")
    
    print(f"📅 Janela: {min_date.date()} ate {max_date.date()}")
    features, dates = build_feature_tensor(nodes_gdf, occ_df, min_date, max_date)
    coords = np.array(list(zip(nodes_proj.geometry.x, nodes_proj.geometry.y)))
    adj_geo = (cdist(coords, coords) <= 3000).astype(float)
    adj_conf = np.eye(len(nodes_gdf))
    p_dir = 'data/processed'
    save_regional_dataset(nodes_gdf, features, adj_geo, adj_conf, dates, 'fortaleza', os.path.join(p_dir, 'processed_fortaleza.pkl'))
    save_regional_dataset(nodes_gdf, features, adj_geo, adj_conf, dates, 'rmf', os.path.join(p_dir, 'processed_rmf.pkl'))
    save_regional_dataset(nodes_gdf, features, adj_geo, adj_conf, dates, 'interior', os.path.join(p_dir, 'processed_interior.pkl'))
    save_regional_dataset(nodes_gdf, features, adj_geo, adj_conf, dates, 'global', os.path.join(p_dir, 'processed_graph_data_global.pkl'))

if __name__ == "__main__":
    main()

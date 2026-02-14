import json
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point
import unicodedata

# Configuração Phase 5 Final (LABORATÓRIO FORTALEZA 121)
DATA_DIR = 'data/raw'
INTELIGENCIA_DIR = os.path.join(DATA_DIR, 'inteligencia')
BAIRROS_FILE = os.path.join(DATA_DIR, 'bairros_centros_latlong.json')
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'View_Ocorrencias_2022_ENRIQUECIDO.csv')
EXOGENOUS_FILE = 'data/exogenous_events.json' 
OUTPUT_FILE = 'data/processed/processed_graph_data.pkl'

# Thresholds
NEIGHBOR_DIST_METERS = 2000 
FACTION_ASSIGN_DIST_METERS = 1000 

def normalize_text(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def load_nodes_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    TO_REMOVE = ['ALTO ALEGRE II', 'CIDADE NOVA', 'DIF III', 'GUADALAJARA', 'INDUSTRIAL', 'IPARANA', 'MARECHAL RONDON', 'PARQUE ALBANO', 'PARQUE DAS NAÇÕES', 'PARQUE LEBLON', 'PARQUE SOLEDADE', 'PRECABURA', 'RACHEL DE QUEIROZ', 'TABAPUÁ', 'URUCUTUBA', 'PARQUE DAS NACOES']
    MERGE_MAP = {'CONJUNTO CEARÁ I': 'CONJUNTO CEARÁ', 'CONJUNTO CEARÁ II': 'CONJUNTO CEARÁ', 'PRAIA DO FUTURO I': 'PRAIA DO FUTURO', 'PRAIA DO FUTURO II': 'PRAIA DO FUTURO'}

    for name, info in data.items():
        norm_name = name.upper().strip()
        if norm_name in TO_REMOVE or info.get('regiao', '').lower() != 'fortaleza': continue
        final_name = MERGE_MAP.get(norm_name, norm_name)
        records.append({'name': final_name, 'latitude': info['lat'], 'longitude': info['long'], 'regiao': 'fortaleza'})
    
    df = pd.DataFrame(records).groupby('name').agg({'latitude': 'mean', 'longitude': 'mean', 'regiao': 'first'}).reset_index()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    return gdf, gdf.to_crs(epsg=3857)

def load_polygon_cache(filepath):
    cache = {}
    if not os.path.exists(filepath): return cache
    try:
        gdf = gpd.read_file(filepath)
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        else: gdf = gdf.to_crs(epsg=4326)
        name_col = None
        for col in ['name', 'NAME', 'nome', 'NOME', 'NM_MUN', 'NM_BAIRRO', 'bairro', 'municipio']:
            if col in gdf.columns:
                name_col = col
                break
        if name_col:
            for idx, row in gdf.iterrows():
                name = normalize_text(str(row[name_col]))
                if name: cache[name] = row.geometry
    except Exception: pass
    return cache

def enrich_node_geometries(nodes_gdf):
    print("Enriquecendo geometrias...")
    bairro_file = os.path.join(DATA_DIR, 'fortaleza_bairros.geojson')
    cache = load_polygon_cache(bairro_file)
    new_geoms = []
    for idx, row in nodes_gdf.iterrows():
        name = normalize_text(row['name'])
        new_geoms.append(cache.get(name, row.geometry))
    nodes_gdf['geometry'] = new_geoms
    return nodes_gdf

def load_faction_layers(directory):
    layers = {}
    files = {'CV': 'COMANDO VERMELHO.geojson', 'TCP': 'TERCEIRO COMANDO PURO.geojson', 'GHOST': 'TERRITÓRIOS FANTASMAS.geojson', 'DISPUTA': 'COMUNIDADES EM DISPUTA.geojson'}
    for key, filename in files.items():
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
                layers[key] = gdf.to_crs(epsg=3857)
            except Exception: layers[key] = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
        else: layers[key] = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    return layers

def calculate_tension_index(nodes_proj, layers):
    tension_values = []
    PROXIMITY_TOLERANCE = 300.0
    disputa_union = layers['DISPUTA'].union_all() if not layers['DISPUTA'].empty else None
    ghost_union = layers['GHOST'].union_all() if not layers['GHOST'].empty else None
    for idx, row in nodes_proj.iterrows():
        point = row.geometry
        val = 0.0
        if disputa_union and (point.intersects(disputa_union) or point.distance(disputa_union) < PROXIMITY_TOLERANCE): val = 1.0
        elif ghost_union and (point.intersects(ghost_union) or point.distance(ghost_union) < PROXIMITY_TOLERANCE): val = 0.5
        tension_values.append(val)
    return np.array(tension_values)

def calculate_faction_assignment(nodes_proj, layers):
    assigned_factions = []
    for idx, row in nodes_proj.iterrows():
        point = row.geometry
        d_cv = layers['CV'].distance(point).min() if not layers['CV'].empty else 99999.0
        d_tcp = layers['TCP'].distance(point).min() if not layers['TCP'].empty else 99999.0
        if d_cv < d_tcp and d_cv < FACTION_ASSIGN_DIST_METERS: assigned_factions.append('CV')
        elif d_tcp <= d_cv and d_tcp < FACTION_ASSIGN_DIST_METERS: assigned_factions.append('TCP')
        else: assigned_factions.append('NEUTRO')
    return np.array(assigned_factions)

def load_and_assign_occurrences_csv(filepath, nodes_gdf):
    node_name_map = {normalize_text(name): idx for idx, name in enumerate(nodes_gdf['name'])}
    df = pd.read_csv(filepath)
    col_map = {'Natureza': 'tipo_evento', 'Data': 'data', 'BairroOcor': 'bairro', 'CidadeOcor': 'municipio', 'lat': 'latitude', 'long': 'longitude'}
    df = df.rename(columns=col_map)
    df['municipio_norm'] = df['municipio'].astype(str).apply(normalize_text)
    df['bairro_norm'] = df['bairro'].astype(str).apply(normalize_text)
    MERGE_OCC = {'CONJUNTO CEARÁ I': 'CONJUNTO CEARÁ', 'CONJUNTO CEARÁ II': 'CONJUNTO CEARÁ', 'PRAIA DO FUTURO I': 'PRAIA DO FUTURO', 'PRAIA DO FUTURO II': 'PRAIA DO FUTURO'}
    df['bairro_norm'] = df['bairro_norm'].replace(MERGE_OCC)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    nodes_coords = list(zip(nodes_gdf.geometry.y, nodes_gdf.geometry.x))
    tree = KDTree(nodes_coords)
    def assign_node(row):
        bairro = row['bairro_norm']
        if bairro in node_name_map: return node_name_map[bairro]
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            dist, idx = tree.query((row['latitude'], row['longitude']))
            if dist < 0.02: return int(idx)
        return -1
    df['node_id'] = df.apply(assign_node, axis=1)
    return df[(df['node_id'] != -1) & (df['data'].notna())].copy()

def build_feature_tensor(nodes_gdf, occurrences_df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_nodes, num_timesteps = len(nodes_gdf), len(date_range)
    features = np.zeros((num_nodes, num_timesteps, 29), dtype=np.float32)

    occurrences_df['tipo_upper'] = occurrences_df['tipo_evento'].astype(str).str.upper()
    
    # Canal 0: CVLI Real e Letalidade por Arma de Fogo (ANTIRRUÍDO)
    is_cvli = (
        occurrences_df['tipo_upper'].str.contains('HOMICIDIO') | 
        occurrences_df['tipo_upper'].str.contains('FEMINICIDIO') | 
        occurrences_df['tipo_upper'].str.contains('LATROCINIO') | 
        occurrences_df['tipo_upper'].str.contains('MORTE') |
        occurrences_df['tipo_upper'].str.contains('INTERVENÇÃO POLICIAL LETAL') |
        (occurrences_df['tipo_upper'].str.contains('LESÃO CORPORAL') & 
         (occurrences_df['tipo_upper'].str.contains('BALA') | occurrences_df['tipo_upper'].str.contains('FOGO'))) |
        (occurrences_df['tipo_upper'].str.contains('TENTATIVA') & 
         (occurrences_df['tipo_upper'].str.contains('HOMICIDIO') | occurrences_df['tipo_upper'].str.contains('BALA')))
    )
    
    is_veiculo = (occurrences_df['tipo_upper'].str.contains('ROUBO') | occurrences_df['tipo_upper'].str.contains('FURTO')) & (occurrences_df['tipo_upper'].str.contains('VEÍCULO') | occurrences_df['tipo_upper'].str.contains('CARRO') | occurrences_df['tipo_upper'].str.contains('MOTO'))

    df_v = occurrences_df[(occurrences_df['data'] >= start_date) & (occurrences_df['data'] <= end_date)].copy()
    df_v['day_idx'] = (df_v['data'] - start_date).dt.days

    # 1. Atribuição de Ocorrências
    for node, day in df_v[is_cvli].groupby(['node_id', 'day_idx']).groups:
        features[node, day, 0] += 1
    for node, day in df_v[is_veiculo].groupby(['node_id', 'day_idx']).groups:
        features[node, day, 1] += 1

    # 2. CITY PULSE (Temperatura da Cidade)
    for d_idx in range(num_timesteps):
        daily_total = features[:, d_idx, 0].sum()
        features[:, d_idx, 28] = daily_total 

    # 3. Inteligência (Incursão e Choques)
    factions = nodes_gdf['faction'].values
    RIVALS_MAP = {'CV': ['TCP', 'GDE', 'PCC'], 'TCP': ['CV'], 'GDE': ['CV'], 'PCC': ['CV']}
    for d_idx in range(num_timesteps):
        nodes_with_theft = np.where(features[:, d_idx, 1] > 0)[0]
        for n_idx in nodes_with_theft:
            for r_fac in RIVALS_MAP.get(factions[n_idx], []):
                features[np.where(factions == r_fac)[0], d_idx, 26] = 1.0

    # Choque de Instabilidade (Expulsões)
    if os.path.exists(EXOGENOUS_FILE):
        try:
            with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
                exo_data = json.load(f)
            for evt in exo_data:
                evt_date = pd.to_datetime(evt['date'])
                if start_date <= evt_date <= end_date:
                    d_idx = (evt_date - start_date).days
                    description = evt.get('description', '').upper()
                    intensity = 2.0 if any(w in description for w in ['EXPULSÃO', 'DESLOCAMENTO']) else 1.0
                    for day_offset in range(7):
                        if d_idx + day_offset < num_timesteps:
                            target_fac = evt.get('faction', 'ALL')
                            if target_fac == 'ALL': features[:, d_idx + day_offset, 25] = intensity
                            else: features[np.where(factions == target_fac)[0], d_idx + day_offset, 25] = intensity
        except Exception: pass

    # Sazonalidade e Tensão
    for d_idx, date in enumerate(date_range):
        features[:, d_idx, 3 + date.weekday()] = 1.0
        features[:, d_idx, 10 + date.month - 1] = 1.0
        if date.weekday() >= 5: features[:, d_idx, 22] = 1.0
            
    for n in range(num_nodes):
        features[n, :, 24] = pd.Series(features[n, :, 0]).rolling(window=7, min_periods=1).mean().values

    features[:, :, 2] = nodes_gdf['tension_index'].values[:, np.newaxis]
    features[:, :, 27] = 1.0 
    return features, date_range

def main():
    print("Iniciando Processamento v6.1 (CENTRO NOISE REDUCTION)...")
    nodes_gdf, nodes_proj = load_nodes_from_json(BAIRROS_FILE)
    nodes_gdf = enrich_node_geometries(nodes_gdf)
    layers = load_faction_layers(INTELIGENCIA_DIR)
    nodes_gdf['tension_index'] = calculate_tension_index(nodes_proj, layers)
    nodes_gdf['faction'] = calculate_faction_assignment(nodes_proj, layers)
    occurrences_df = load_and_assign_occurrences_csv(OCORRENCIAS_FILE, nodes_gdf)
    max_date = occurrences_df['data'].max()
    min_date = max_date - pd.Timedelta(days=1000)
    node_features, dates = build_feature_tensor(nodes_gdf, occurrences_df, min_date, max_date)
    
    from scipy.spatial.distance import cdist
    coords = np.array(list(zip(nodes_proj.geometry.x, nodes_proj.geometry.y)))
    adj_geo = (cdist(coords, coords) <= NEIGHBOR_DIST_METERS).astype(float)
    adj_conf = np.zeros_like(adj_geo)
    factions = nodes_gdf['faction'].values
    RIVALS = {frozenset(['CV', 'TCP']), frozenset(['CV', 'GDE']), frozenset(['CV', 'PCC'])}
    for i in range(len(nodes_gdf)):
        for j in range(len(nodes_gdf)):
            if i==j: adj_conf[i,j]=1.0; continue
            if factions[i] == factions[j] and factions[i] != 'NEUTRO': adj_conf[i,j]=1.0
            elif frozenset([factions[i], factions[j]]) in RIVALS: adj_conf[i,j]=1.0

    data_pack = {
        'node_features': node_features, 'adj_geo': adj_geo, 'adj_conflict': adj_conf,
        'dates': dates, 'nodes_gdf': nodes_gdf,
        'feature_names': ['CVLI', 'VEHICLE', 'TENSION', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'WEEKEND', 'HOLIDAY', 'MOMENTUM_7D', 'INTEL_SHOCK', 'INCURSION', 'URBAN', 'CITY_PULSE']
    }
    with open(OUTPUT_FILE, 'wb') as f: pickle.dump(data_pack, f)
    print(f"Sucesso! Dados limpos (Antirruído Centro) gerados.")

if __name__ == "__main__":
    main()

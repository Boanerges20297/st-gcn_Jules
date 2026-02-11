import json
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point
import unicodedata

# Configuração
DATA_DIR = 'data/raw'
INTELIGENCIA_DIR = os.path.join(DATA_DIR, 'inteligencia')
BAIRROS_FILE = os.path.join(DATA_DIR, 'bairros_centros_latlong.json')
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais.json')
OUTPUT_FILE = 'data/processed/processed_graph_data.pkl'

# Thresholds
NEIGHBOR_DIST_METERS = 2000  # Distância para considerar vizinho geográfico
FACTION_ASSIGN_DIST_METERS = 1000 # Distância máxima para atribuir facção a um bairro

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def load_polygon_cache(filepath):
    """Carrega GeoJSON e retorna dict {normalized_name: geometry}."""
    cache = {}
    if not os.path.exists(filepath):
        return cache

    try:
        gdf = gpd.read_file(filepath)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)

        # Identificar coluna de nome
        name_col = None
        for col in ['name', 'NAME', 'nome', 'NOME', 'NM_MUN', 'NM_BAIRRO', 'bairro', 'municipio']:
            if col in gdf.columns:
                name_col = col
                break

        if name_col:
            for idx, row in gdf.iterrows():
                name = normalize_text(str(row[name_col]))
                if name:
                    cache[name] = row.geometry
    except Exception:
        pass

    return cache

def enrich_node_geometries(nodes_gdf):
    """Tenta substituir Ponto por Polígono usando arquivos externos."""
    print("Enriquecendo geometrias dos nós (Merge Polygons)...")

    mun_file = os.path.join(DATA_DIR, 'ceara_municipios.geojson')
    bairro_file = os.path.join(DATA_DIR, 'fortaleza_bairros.geojson')
    interior_file = os.path.join(DATA_DIR, 'ceara_interior.geojson') 

    caches = []
    caches.append(load_polygon_cache(mun_file))
    caches.append(load_polygon_cache(bairro_file))
    caches.append(load_polygon_cache(interior_file))

    count_merged = 0
    new_geoms = []

    for idx, row in nodes_gdf.iterrows():
        name = normalize_text(row['name'])
        poly = None

        for cache in caches:
            if name in cache:
                poly = cache[name]
                break

        if poly is not None:
            new_geoms.append(poly)
            count_merged += 1
        else:
            new_geoms.append(row.geometry)

    nodes_gdf['geometry'] = new_geoms
    print(f"Polígonos atribuídos: {count_merged}/{len(nodes_gdf)}")

    if count_merged == 0:
        print("\n!!! ALERTA: Nenhum nó recebeu geometria de polígono. O mapa exibirá apenas pontos.")
        print("    Certifique-se de que os arquivos .geojson estão em data/raw/.\n")

    return nodes_gdf

def load_nodes_from_json(filepath):
    """Carrega os bairros do JSON e converte para GeoDataFrame (Centróides)."""
    print(f"Carregando nós de {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for name, info in data.items():
        if name in ["Nome", "null", "None", ""] or name is None:
            continue

        regiao = info.get('regiao', 'desconhecido').lower()
        node_type = 'bairro' if regiao in ['fortaleza', 'rmf'] else 'cidade'

        records.append({
            'name': name,
            'latitude': info['lat'],
            'longitude': info['long'],
            'regiao': regiao,
            'node_type': node_type
        })
    
    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    # Projetar para métrico
    gdf_proj = gdf.to_crs(epsg=3857)
    print(f"Total de nós carregados: {len(gdf)}")
    return gdf, gdf_proj

def load_faction_layers(directory):
    """Carrega GeoJSONs das facções."""
    layers = {}
    print(f"Carregando camadas de inteligência de {directory}...")

    files = {
        'CV': 'COMANDO VERMELHO.geojson',
        'TCP': 'TERCEIRO COMANDO PURO.geojson',
        'GHOST': 'TERRITÓRIOS FANTASMAS.geojson',
        'DISPUTA': 'COMUNIDADES EM DISPUTA.geojson'
    }

    for key, filename in files.items():
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                layers[key] = gdf.to_crs(epsg=3857) # Métrico
                print(f"  - {key}: {len(gdf)} geometrias carregadas.")
            except Exception as e:
                print(f"Erro ao ler {filename}: {e}")
                layers[key] = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
        else:
            print(f"Arquivo não encontrado: {path}")
            layers[key] = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    return layers

def calculate_tension_index(nodes_proj, layers):
    """Calcula o Índice de Tensão (Canal 2)."""
    print("Calculando Índice de Tensão...")
    tension_values = []
    PROXIMITY_TOLERANCE = 300.0

    disputa_union = layers['DISPUTA'].union_all() if not layers['DISPUTA'].empty else None
    ghost_union = layers['GHOST'].union_all() if not layers['GHOST'].empty else None

    for idx, row in nodes_proj.iterrows():
        point = row.geometry
        val = 0.0

        if disputa_union and (point.intersects(disputa_union) or point.distance(disputa_union) < PROXIMITY_TOLERANCE):
            val = 1.0
        elif ghost_union and (point.intersects(ghost_union) or point.distance(ghost_union) < PROXIMITY_TOLERANCE):
            val = 0.5

        tension_values.append(val)

    return np.array(tension_values)

def calculate_faction_assignment(nodes_proj, layers):
    """Define a facção dominante do nó."""
    print("Atribuindo facções aos nós...")
    assigned_factions = []

    for idx, row in nodes_proj.iterrows():
        point = row.geometry
        
        d_cv = layers['CV'].distance(point).min() if not layers['CV'].empty else 99999.0
        d_tcp = layers['TCP'].distance(point).min() if not layers['TCP'].empty else 99999.0

        if d_cv < d_tcp and d_cv < FACTION_ASSIGN_DIST_METERS:
            assigned_factions.append('CV')
        elif d_tcp <= d_cv and d_tcp < FACTION_ASSIGN_DIST_METERS:
            assigned_factions.append('TCP')
        else:
            assigned_factions.append('NEUTRO')

    return np.array(assigned_factions)

def load_and_assign_occurrences(filepath, nodes_gdf):
    """
    Carrega ocorrências e as atribui aos nós com tratamento ROBUSTO de datas e listas.
    Resolve o problema de perda de dados históricos.
    """
    print(f"Processando ocorrências de {filepath}...")

    # Identificar índice do Barroso para NLP fix
    barroso_indices = nodes_gdf.index[nodes_gdf['name'] == 'BARROSO'].tolist()
    barroso_idx = barroso_indices[0] if barroso_indices else None

    # Mapeamento reverso para lookup rápido
    node_name_map = {normalize_text(name): idx for idx, name in enumerate(nodes_gdf['name'])}
    interior_nodes = set(nodes_gdf[nodes_gdf['node_type'] == 'cidade'].index)
    capital_rmf_nodes = set(nodes_gdf[nodes_gdf['node_type'] == 'bairro'].index)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    occurrences_list = []
    # Lógica robusta para extrair lista plana de ocorrências
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'data' in item and isinstance(item['data'], list):
                # Caso: Lista de blocos com chave 'data'
                occurrences_list.extend(item['data'])
            elif isinstance(item, dict):
                # Caso: Lista direta de objetos
                occurrences_list.append(item)
    elif isinstance(data, dict) and 'data' in data:
         occurrences_list = data['data']
        
    print(f"Total bruto de registros encontrados no JSON: {len(occurrences_list)}")

    df = pd.DataFrame(occurrences_list)
    df = df.rename(columns={'Natureza': 'tipo_evento'})

    # Normalizações básicas
    if 'municipio' not in df.columns: df['municipio'] = ''
    if 'bairro' not in df.columns: df['bairro'] = ''
    
    df['municipio_norm'] = df['municipio'].astype(str).apply(normalize_text)
    df['bairro_norm'] = df['bairro'].astype(str).apply(normalize_text)
    
    # NLP Search Text
    text_cols = [c for c in df.columns if c.lower() in ['descricao', 'localizacao', 'endereco', 'bairro', 'municipio']]
    df['search_text'] = df[text_cols].fillna('').apply(lambda x: ' '.join(x.astype(str)).upper(), axis=1)

    # Coordenadas
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Prepara KDTree
    valid_coords_mask = df['latitude'].notna() & df['longitude'].notna()
    nodes_coords = list(zip(nodes_gdf.geometry.y, nodes_gdf.geometry.x))
    tree = KDTree(nodes_coords)

    print("Iniciando atribuição lógica (Nome vs Espacial)...")

    def assign_node(row):
        # NLP Fix Barroso
        if barroso_idx is not None and 'BARROSO' in row['search_text']:
            return barroso_idx, 'nlp_barroso'

        mun = row['municipio_norm']
        bairro = row['bairro_norm']

        # Prioridade 1: Match Nome Interior
        if mun in node_name_map:
            idx = node_name_map[mun]
            if idx in interior_nodes:
                return idx, 'name_city'

        # Prioridade 2: Match Nome Capital/RMF
        if bairro in node_name_map:
            idx = node_name_map[bairro]
            if idx in capital_rmf_nodes:
                return idx, 'name_bairro'

        # Prioridade 3: Espacial
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            d, idx = tree.query((row['latitude'], row['longitude']))
            return int(idx), 'spatial'

        return -1, 'unassigned'

    results = df.apply(assign_node, axis=1)
    df['node_id'] = results.apply(lambda x: x[0])
    df['match_type'] = results.apply(lambda x: x[1])

    print("Estatísticas de Atribuição (Pré-filtro):")
    print(df['match_type'].value_counts())

    # --- TRATAMENTO ROBUSTO DE DATAS ---
    if 'data' in df.columns:
        # 1. Tenta converter assumindo formato ISO (YYYY-MM-DD) ou misto
        df['data_iso'] = pd.to_datetime(df['data'], errors='coerce')
        
        # 2. Para o que falhou (NaT), tenta assumir dia primeiro (DD/MM/YYYY)
        mask_fail = df['data_iso'].isna()
        if mask_fail.any():
            print(f"Tentando recuperar {mask_fail.sum()} datas com formato brasileiro...")
            df.loc[mask_fail, 'data_iso'] = pd.to_datetime(df.loc[mask_fail, 'data'], dayfirst=True, errors='coerce')
        
        df['data'] = df['data_iso']

    # Filtragem Final
    df_final = df[ (df['node_id'] != -1) & (df['data'].notna()) ].copy()
    
    print(f"Registros descartados por falha na Data: {df['data'].isna().sum()}")
    print(f"Registros descartados por falha no Local: {(df['node_id'] == -1).sum()}")
    print(f"Total de ocorrências VÁLIDAS para o modelo: {len(df_final)}")
    
    if len(df_final) > 0:
        print(f"Histórico disponível: de {df_final['data'].min()} a {df_final['data'].max()}")

    return df_final

def build_feature_tensor(nodes_gdf, occurrences_df, start_date, end_date, exogenous_events=None):
    """Constrói o tensor (Nodes, Time, 26)."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_nodes = len(nodes_gdf)
    num_timesteps = len(date_range)
    num_features = 26

    features = np.zeros((num_nodes, num_timesteps, num_features), dtype=np.float32)

    # Filtros
    cvli_types = ['HOMICIDIO DOLOSO', 'FEMINICÍDIO', 'ROUBO SEGUIDO DE MORTE (LATROCINIO)', 'LESAO CORPORAL SEGUIDA DE MORTE']
    is_cvli = occurrences_df['tipo_evento'].astype(str).str.upper().isin(cvli_types)

    tipo_upper = occurrences_df['tipo_evento'].astype(str).str.upper()
    is_cvp = tipo_upper.str.contains('ROUBO') | tipo_upper.str.contains('FURTO')

    occurrences_df['day_idx'] = (occurrences_df['data'] - start_date).dt.days
    valid_mask = (occurrences_df['day_idx'] >= 0) & (occurrences_df['day_idx'] < num_timesteps)
    df_valid = occurrences_df[valid_mask]

    print("Agregando 26 Canais...")

    # Channel 0: CVLI
    cvli_counts = df_valid[is_cvli[valid_mask]].groupby(['node_id', 'day_idx']).size()
    for (node, day), count in cvli_counts.items():
        features[node, day, 0] = count

    # Channel 1: CVP
    cvp_counts = df_valid[is_cvp[valid_mask]].groupby(['node_id', 'day_idx']).size()
    for (node, day), count in cvp_counts.items():
        features[node, day, 1] = count

    # Channel 2: Tension
    tension_values = nodes_gdf['tension_index'].values
    features[:, :, 2] = np.tile(tension_values[:, np.newaxis], (1, num_timesteps))

    # Channels 3-9: DOW
    for day_idx, date in enumerate(date_range):
        dow = date.weekday()
        features[:, day_idx, 3 + dow] = 1.0
    
    # Channels 10-21: Month
    for day_idx, date in enumerate(date_range):
        month = date.month - 1
        features[:, day_idx, 10 + month] = 1.0
    
    # Channel 22: Weekend
    for day_idx, date in enumerate(date_range):
        if date.weekday() >= 5:
            features[:, day_idx, 22] = 1.0

    print("[OK] Features categoricas geradas (26 canais)")
    return features, date_range

def create_adjacency_matrices(nodes_gdf, nodes_proj):
    """
    Cria matrizes adj_geo (Física) e adj_semantic (Lógica/Facção).
    Removemos a restrição geográfica da matriz de conflito.
    """
    n = len(nodes_gdf)
    adj_geo = np.zeros((n, n), dtype=float)
    adj_semantic = np.zeros((n, n), dtype=float) 
    
    coords = np.array(list(zip(nodes_proj.geometry.x, nodes_proj.geometry.y)))
    factions = nodes_gdf['faction'].values
    
    print("Calculando matrizes de adjacência (Geo + Semântica)...")
    
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords) # metros
    
    # 1. Matriz Geográfica
    mask_geo = dists <= NEIGHBOR_DIST_METERS
    adj_geo[mask_geo] = 1.0
    np.fill_diagonal(adj_geo, 1.0)
    
    # 2. Matriz Semântica (A Nuvem)
    RIVALS = {
        frozenset(['CV', 'TCP']),
        frozenset(['CV', 'GDE']),
        frozenset(['CV', 'PCC']),
        frozenset(['GDE', 'PCC'])
    }
    
    count_allies = 0
    count_enemies = 0

    for i in range(n):
        for j in range(n):
            if i == j: 
                adj_semantic[i, j] = 1.0 
                continue

            f_i = str(factions[i]).upper()
            f_j = str(factions[j]).upper()
            
            if f_i in ['NEUTRO', 'N/A', 'NONE'] or f_j in ['NEUTRO', 'N/A', 'NONE']:
                continue

            # REGRA 1: Logística (Aliados)
            if f_i == f_j:
                adj_semantic[i, j] = 1.0
                count_allies += 1

            # REGRA 2: Guerra (Rivais - SEM restrição geográfica)
            elif frozenset([f_i, f_j]) in RIVALS:
                adj_semantic[i, j] = 1.0 
                count_enemies += 1
    
    print(f"  [Grafo Semântico] Conexões de Aliança criadas: {count_allies}")
    print(f"  [Grafo Semântico] Conexões de Guerra criadas: {count_enemies}")
    
    return adj_geo, adj_semantic

def main():
    print("Iniciando Pipeline de Dados v3 (Paradigm Shift)...")
    
    # 1. Carregar Nós
    nodes_gdf, nodes_proj = load_nodes_from_json(BAIRROS_FILE)
    
    # 1.1 Merge Polygons
    nodes_gdf = enrich_node_geometries(nodes_gdf)
    nodes_proj = nodes_gdf.to_crs(epsg=3857)

    # 2. Carregar Camadas
    layers = load_faction_layers(INTELIGENCIA_DIR)
    
    # 3. Calcular Tensão e Facção
    nodes_gdf['tension_index'] = calculate_tension_index(nodes_proj, layers)
    nodes_gdf['faction'] = calculate_faction_assignment(nodes_proj, layers)
    
    # 4. Carregar Ocorrências (Agora robusto)
    occurrences_df = load_and_assign_occurrences(OCORRENCIAS_FILE, nodes_gdf)

    # 4.5. Carregar Eventos Exógenos
    exogenous_events = []
    exogenous_file = os.path.join('data', 'exogenous_events.json')
    if os.path.exists(exogenous_file):
        try:
            with open(exogenous_file, 'r', encoding='utf-8') as f:
                exogenous_events = json.load(f)
            print(f"Carregados {len(exogenous_events)} lotes de eventos exógenos")
        except Exception as e:
            print(f"Erro ao carregar eventos exógenos: {e}")
            exogenous_events = []

    # 5. Feature Tensor
    max_date = occurrences_df['data'].max()
    # Pega 6 meses de histórico para permitir Backtesting/Validação
    min_date = max_date - pd.Timedelta(days=180)
    print(f"Janela de Treino Definida: {min_date.date()} a {max_date.date()}")

    node_features, dates = build_feature_tensor(nodes_gdf, occurrences_df, min_date, max_date, exogenous_events)

    # 6. Grafos
    adj_geo, adj_conflict = create_adjacency_matrices(nodes_gdf, nodes_proj)

    # 7. Salvar
    data_pack = {
        'node_features': node_features,
        'adj_geo': adj_geo,
        'adj_faction': adj_conflict, # CHAVE NOVA PARA APP.PY
        'adj_conflict': adj_conflict,
        'dates': dates,
        'nodes_gdf': nodes_gdf,
        'feature_names': [
            'CVLI', 'CVP', 'TENSION_INDEX',
            'DOW_MON', 'DOW_TUE', 'DOW_WED', 'DOW_THU', 'DOW_FRI', 'DOW_SAT', 'DOW_SUN',
            'MONTH_JAN', 'MONTH_FEB', 'MONTH_MAR', 'MONTH_APR', 'MONTH_MAY', 'MONTH_JUN',
            'MONTH_JUL', 'MONTH_AUG', 'MONTH_SEP', 'MONTH_OCT', 'MONTH_NOV', 'MONTH_DEC',
            'IS_WEEKEND', 'RESERVED_23', 'RESERVED_24', 'RESERVED_25'
        ]
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save nodes_gdf separadamente
    nodes_pkl_dir = os.path.join(os.path.dirname(OUTPUT_FILE), 'graph_data')
    os.makedirs(nodes_pkl_dir, exist_ok=True)
    with open(os.path.join(nodes_pkl_dir, 'nodes_gdf.pkl'), 'wb') as f:
        pickle.dump(nodes_gdf, f)

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data_pack, f)
        
    print(f"[OK] Sucesso! Dados salvos em {OUTPUT_FILE}")
    print(f"  - Shape: {node_features.shape}")

if __name__ == "__main__":
    main()
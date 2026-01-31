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

        # Define o tipo de nó baseada na região
        # Capital/RMF -> Bairro
        # Interior -> Cidade
        node_type = 'bairro' if regiao in ['fortaleza', 'rmf'] else 'cidade'

        records.append({
            'name': name, # Nome do Bairro ou da Cidade
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
    # Projetar para métrico para cálculos de distância
    gdf_proj = gdf.to_crs(epsg=3857)
    print(f"Total de nós carregados: {len(gdf)}")
    print(f"Distribuição de Tipos: {df['node_type'].value_counts()}")
    return gdf, gdf_proj

def load_faction_layers(directory):
    """Carrega GeoJSONs das facções e territórios relevantes."""
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
    """
    Calcula o Índice de Tensão (Canal 2).
    """
    print("Calculando Índice de Tensão...")
    tension_values = []
    PROXIMITY_TOLERANCE = 300.0

    disputa_union = layers['DISPUTA'].union_all() if not layers['DISPUTA'].empty else None
    ghost_union = layers['GHOST'].union_all() if not layers['GHOST'].empty else None

    for idx, row in nodes_proj.iterrows():
        point = row.geometry
        val = 0.0

        # 1.0: Disputa Ativa
        if disputa_union and (point.intersects(disputa_union) or point.distance(disputa_union) < PROXIMITY_TOLERANCE):
            val = 1.0
        # 0.5: Ghost ou Fronteira
        elif ghost_union and (point.intersects(ghost_union) or point.distance(ghost_union) < PROXIMITY_TOLERANCE):
            val = 0.5

        tension_values.append(val)

    return np.array(tension_values)

def calculate_faction_assignment(nodes_proj, layers):
    """
    Define a facção dominante do nó.
    """
    print("Atribuindo facções aos nós...")
    assigned_factions = []

    for idx, row in nodes_proj.iterrows():
        point = row.geometry
        
        if not layers['CV'].empty:
            d_cv = layers['CV'].distance(point).min()
        else:
            d_cv = 99999.0
        
        if not layers['TCP'].empty:
            d_tcp = layers['TCP'].distance(point).min()
        else:
            d_tcp = 99999.0

        if d_cv < d_tcp and d_cv < FACTION_ASSIGN_DIST_METERS:
            assigned_factions.append('CV')
        elif d_tcp <= d_cv and d_tcp < FACTION_ASSIGN_DIST_METERS:
            assigned_factions.append('TCP')
        else:
            assigned_factions.append('NEUTRO')

    return np.array(assigned_factions)

def load_and_assign_occurrences(filepath, nodes_gdf):
    """
    Carrega ocorrências e as atribui aos nós respeitando a lógica Capital/RMF vs Interior.
    """
    print(f"Processando ocorrências de {filepath}...")

    # Identificar índice do Barroso para NLP fix
    barroso_indices = nodes_gdf.index[nodes_gdf['name'] == 'BARROSO'].tolist()
    barroso_idx = barroso_indices[0] if barroso_indices else None

    # Mapeamento reverso para lookup rápido
    # Normalizamos as chaves para uppercase ascii
    node_name_map = {normalize_text(name): idx for idx, name in enumerate(nodes_gdf['name'])}

    # Mapas separados para conflitos de nome (se houver Bairro com nome de Cidade?)
    # A estrutura atual já garante unicidade de nós no JSON (chaves únicas).

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    occurrences_list = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'data' in item and isinstance(item['data'], list):
                occurrences_list = item['data']
                break
    if not occurrences_list and isinstance(data, list) and len(data) > 0:
        occurrences_list = data

    df = pd.DataFrame(occurrences_list)
    df = df.rename(columns={'Natureza': 'tipo_evento'})

    # Normalizar textos
    if 'municipio' not in df.columns: df['municipio'] = ''
    if 'bairro' not in df.columns: df['bairro'] = ''

    # Garantir strings
    df['municipio_norm'] = df['municipio'].astype(str).apply(normalize_text)
    df['bairro_norm'] = df['bairro'].astype(str).apply(normalize_text)

    # NLP Search Text
    text_cols = [c for c in df.columns if c.lower() in ['descricao', 'localizacao', 'endereco', 'bairro', 'municipio']]
    df['search_text'] = df[text_cols].fillna('').apply(lambda x: ' '.join(x.astype(str)).upper(), axis=1)

    # Coordenadas
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Prepara KDTree global para fallback
    valid_coords_mask = df['latitude'].notna() & df['longitude'].notna()
    nodes_coords = list(zip(nodes_gdf.geometry.y, nodes_gdf.geometry.x))
    tree = KDTree(nodes_coords)

    # --- Lógica de Atribuição ---

    # Array para armazenar resultados (-1 = não atribuído)
    assigned_indices = np.full(len(df), -1, dtype=int)

    # 1. Atribuição por Nome (Prioridade)
    # Iterar é lento, vamos tentar vetorizar com map, mas logica condicional complexa.
    # Vamos iterar por segurança lógica neste refactor crítico.

    # Otimização: Criar dicionários de indices de nós por tipo
    interior_nodes = set(nodes_gdf[nodes_gdf['node_type'] == 'cidade'].index)
    capital_rmf_nodes = set(nodes_gdf[nodes_gdf['node_type'] == 'bairro'].index)

    # Contador para estatisticas
    count_name_match = 0
    count_spatial_match = 0
    count_barroso = 0

    print("Iniciando atribuição lógica (Nome vs Espacial)...")

    # Vamos processar em chunks ou apply? Apply row-wise é seguro.
    def assign_node(row):
        # NLP Fix Barroso (Global Override requested previously)
        if barroso_idx is not None and 'BARROSO' in row['search_text']:
            return barroso_idx, 'nlp_barroso'

        mun = row['municipio_norm']
        bairro = row['bairro_norm']

        # Caso Interior: Busca pelo nome do município
        # Como saber se é interior? Lista de municipios da RMF/Capital?
        # A lista de nós já tem essa distinção. Se o nome do município bater com um nó do tipo 'cidade', é isso.

        if mun in node_name_map:
            idx = node_name_map[mun]
            if idx in interior_nodes:
                return idx, 'name_city'

        # Caso Capital/RMF: Busca pelo bairro
        # Se municipio for Fortaleza ou RMF (assumimos que se não achou como cidade interior, pode ser RMF)
        # Tenta match do bairro
        if bairro in node_name_map:
            idx = node_name_map[bairro]
            if idx in capital_rmf_nodes:
                return idx, 'name_bairro'

        # Fallback Espacial
        # Só se tiver coord
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            # Query KDTree (retorna distancia, index)
            # tree.query aceita single point? Sim.
            d, idx = tree.query((row['latitude'], row['longitude']))
            return int(idx), 'spatial'

        return -1, 'unassigned'

    # Aplicar lógica
    results = df.apply(assign_node, axis=1)

    # Desempacotar
    df['node_id'] = results.apply(lambda x: x[0])
    df['match_type'] = results.apply(lambda x: x[1])

    print("Estatísticas de Atribuição:")
    print(df['match_type'].value_counts())

    # Filtrar não atribuídos
    df = df[df['node_id'] != -1]

    # Converter data
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df.dropna(subset=['data'])

    print(f"Total de ocorrências atribuídas: {len(df)}")
    return df

def build_feature_tensor(nodes_gdf, occurrences_df, start_date, end_date):
    """
    Constrói o tensor (Nodes, Time, 3).
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_nodes = len(nodes_gdf)
    num_timesteps = len(date_range)
    num_features = 3

    features = np.zeros((num_nodes, num_timesteps, num_features), dtype=np.float32)

    # Filtros
    cvli_types = ['HOMICIDIO DOLOSO', 'FEMINICÍDIO', 'ROUBO SEGUIDO DE MORTE (LATROCINIO)', 'LESAO CORPORAL SEGUIDA DE MORTE']
    is_cvli = occurrences_df['tipo_evento'].astype(str).str.upper().isin(cvli_types)

    tipo_upper = occurrences_df['tipo_evento'].astype(str).str.upper()
    is_cvp = tipo_upper.str.contains('ROUBO') | tipo_upper.str.contains('FURTO') | tipo_upper.str.contains('VEÍCULO') | tipo_upper.str.contains('MOTO')

    occurrences_df['day_idx'] = (occurrences_df['data'] - start_date).dt.days
    valid_mask = (occurrences_df['day_idx'] >= 0) & (occurrences_df['day_idx'] < num_timesteps)
    df_valid = occurrences_df[valid_mask]

    print("Agregando Channels...")

    # Optimizado: Groupby count
    cvli_counts = df_valid[is_cvli[valid_mask]].groupby(['node_id', 'day_idx']).size()
    # Usar add.at para preenchimento rápido se indices fossem numpy, mas aqui é sparse map. Loop ok.
    for (node, day), count in cvli_counts.items():
        features[node, day, 0] = count

    cvp_counts = df_valid[is_cvp[valid_mask]].groupby(['node_id', 'day_idx']).size()
    for (node, day), count in cvp_counts.items():
        features[node, day, 1] = count

    # Channel 2: Tension
    tension_values = nodes_gdf['tension_index'].values
    features[:, :, 2] = np.tile(tension_values[:, np.newaxis], (1, num_timesteps))

    return features, date_range

def create_adjacency_matrices(nodes_gdf, nodes_proj):
    """
    Cria matrizes adj_geo e adj_conflict.
    """
    n = len(nodes_gdf)
    adj_geo = np.zeros((n, n), dtype=float)
    adj_conflict = np.zeros((n, n), dtype=float)
    
    coords = np.array(list(zip(nodes_proj.geometry.x, nodes_proj.geometry.y)))
    factions = nodes_gdf['faction'].values
    
    print("Calculando matrizes de adjacência...")
    
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords) # metros
    
    # Adj Geo
    mask_geo = dists <= NEIGHBOR_DIST_METERS
    adj_geo[mask_geo] = 1.0
    
    # Adj Conflict
    for i in range(n):
        for j in range(n):
            if mask_geo[i, j]:
                f_i = factions[i]
                f_j = factions[j]
                if i == j: continue

                is_rival = (f_i == 'CV' and f_j == 'TCP') or (f_i == 'TCP' and f_j == 'CV')
                if is_rival:
                    adj_conflict[i, j] = 1.0
    
    np.fill_diagonal(adj_geo, 1.0)
    
    return adj_geo, adj_conflict

def main():
    print("Iniciando Pipeline de Dados v3 (Paradigm Shift)...")
    
    # 1. Carregar Nós
    nodes_gdf, nodes_proj = load_nodes_from_json(BAIRROS_FILE)
    
    # 2. Carregar Camadas
    layers = load_faction_layers(INTELIGENCIA_DIR)
    
    # 3. Calcular Tensão e Facção
    nodes_gdf['tension_index'] = calculate_tension_index(nodes_proj, layers)
    nodes_gdf['faction'] = calculate_faction_assignment(nodes_proj, layers)
    
    # 4. Carregar Ocorrências
    occurrences_df = load_and_assign_occurrences(OCORRENCIAS_FILE, nodes_gdf)

    # 5. Feature Tensor
    min_date = occurrences_df['data'].min()
    max_date = occurrences_df['data'].max()
    print(f"Intervalo: {min_date.date()} a {max_date.date()}")

    node_features, dates = build_feature_tensor(nodes_gdf, occurrences_df, min_date, max_date)

    # 6. Grafos
    adj_geo, adj_conflict = create_adjacency_matrices(nodes_gdf, nodes_proj)

    # 7. Salvar
    data_pack = {
        'node_features': node_features,
        'adj_geo': adj_geo,
        'adj_conflict': adj_conflict,
        'nodes_gdf': nodes_gdf,
        'dates': dates,
        'feature_names': ['CVLI', 'CVP_MOVIMENTO', 'TENSION_INDEX']
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data_pack, f)
        
    print(f"✓ Sucesso! Dados salvos em {OUTPUT_FILE}")
    print(f"  - Shape: {node_features.shape}")

if __name__ == "__main__":
    main()

import json
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point

# Configuração
DATA_DIR = 'data/raw'
INTELIGENCIA_DIR = os.path.join(DATA_DIR, 'inteligencia')
BAIRROS_FILE = os.path.join(DATA_DIR, 'bairros_centros_latlong.json')
OCORRENCIAS_FILE = os.path.join(DATA_DIR, 'dados_status_ocorrencias_gerais.json')
OUTPUT_FILE = 'data/processed/processed_graph_data.pkl'

# Thresholds
NEIGHBOR_DIST_METERS = 2000  # Distância para considerar vizinho geográfico
FACTION_ASSIGN_DIST_METERS = 1000 # Distância máxima para atribuir facção a um bairro

def load_nodes_from_json(filepath):
    """Carrega os bairros do JSON e converte para GeoDataFrame (Centróides)."""
    print(f"Carregando nós de {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for name, info in data.items():
        records.append({
            'name': name,
            'latitude': info['lat'],
            'longitude': info['long'],
            'regiao': info.get('regiao', 'desconhecido')
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
    return gdf, gdf_proj

def load_faction_layers(directory):
    """Carrega GeoJSONs das facções e territórios fantasmas."""
    layers = {}
    print(f"Carregando camadas de inteligência de {directory}...")

    files = {
        'CV': 'COMANDO VERMELHO.geojson',
        'TCP': 'TERCEIRO COMANDO PURO.geojson',
        'GHOST': 'TERRITÓRIOS FANTASMAS.geojson'
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

def calculate_distances_and_faction(nodes_proj, layers):
    """
    Calcula distâncias para facções e define a facção dominante do nó.
    Retorna arrays de distância e lista de facções.
    """
    print("Calculando distâncias e atribuindo facções...")
    dist_cv = []
    dist_tcp = []
    dist_ghost = []
    assigned_factions = []

    # Prepara índices espaciais (opcional, mas bom se houver muitos polígonos)
    # Aqui faremos força bruta vetorizada via sjoin_nearest se possível,
    # ou distance() direto já que n_nodes é pequeno (~200).

    for idx, row in nodes_proj.iterrows():
        point = row.geometry
        
        # Distância CV
        if not layers['CV'].empty:
            d_cv = layers['CV'].distance(point).min()
        else:
            d_cv = 99999.0
        dist_cv.append(d_cv)
        
        # Distância TCP
        if not layers['TCP'].empty:
            d_tcp = layers['TCP'].distance(point).min()
        else:
            d_tcp = 99999.0
        dist_tcp.append(d_tcp)

        # Distância Ghost
        if not layers['GHOST'].empty:
            d_ghost = layers['GHOST'].distance(point).min()
        else:
            d_ghost = 99999.0
        dist_ghost.append(d_ghost)

        # Atribuição de Facção (CV, TCP ou NEUTRO)
        # Lógica: O mais próximo, desde que < threshold
        if d_cv < d_tcp and d_cv < FACTION_ASSIGN_DIST_METERS:
            assigned_factions.append('CV')
        elif d_tcp <= d_cv and d_tcp < FACTION_ASSIGN_DIST_METERS:
            assigned_factions.append('TCP')
        else:
            assigned_factions.append('NEUTRO')

    return np.array(dist_cv), np.array(dist_tcp), np.array(dist_ghost), np.array(assigned_factions)

def load_and_assign_occurrences(filepath, nodes_gdf):
    """
    Carrega ocorrências e as atribui ao nó mais próximo (KDTree).
    """
    print(f"Processando ocorrências de {filepath}...")

    # Ler JSON
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extrair lista
    occurrences_list = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'data' in item and isinstance(item['data'], list):
                occurrences_list = item['data']
                break
    if not occurrences_list and isinstance(data, list) and len(data) > 0:
        occurrences_list = data

    df = pd.DataFrame(occurrences_list)

    # Limpeza básica
    df = df.rename(columns={'Natureza': 'tipo_evento'}) # Normalizar se necessário

    # Converter coordenadas
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Converter data
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    else:
        # Tentar reconstruir data se houver colunas separadas? Não, assumir coluna 'data'
        pass
    df = df.dropna(subset=['data'])

    # Construir KDTree dos nós
    nodes_coords = list(zip(nodes_gdf.geometry.y, nodes_gdf.geometry.x)) # lat, lon (KDTree usa cartesiano, mas para proximidade pequena ok, ou converter)
    # Melhor usar coordenadas projetadas ou cartesiano simples?
    # Como os dados brutos estão em lat/lon, vamos usar lat/lon direto para "Nearest Neighbor" aproximado.
    # Para precisão geográfica perfeita deveria projetar, mas para "nearest neighbor" em escala de cidade, lat/lon funciona razoavelmente.
    tree = KDTree(nodes_coords)

    event_coords = list(zip(df['latitude'], df['longitude']))
    distances, indices = tree.query(event_coords)

    # Adicionar node_id ao df
    df['node_id'] = indices
    df['dist_to_node'] = distances

    # Filtrar eventos muito distantes se necessário (ex: > 0.05 graus)
    # df = df[df['dist_to_node'] < 0.05]

    print(f"Total de ocorrências atribuídas: {len(df)}")
    return df

def build_feature_tensor(nodes_gdf, occurrences_df, start_date, end_date):
    """
    Constrói o tensor (Nodes, Time, Features).
    Features:
    0: CVLI Count
    1: ROUBO_VEICULO Count
    2: Distância CV (Normalizada)
    3: Distância TCP (Normalizada)
    4: Distância Ghost (Normalizada)
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_nodes = len(nodes_gdf)
    num_timesteps = len(date_range)
    num_features = 5

    features = np.zeros((num_nodes, num_timesteps, num_features), dtype=np.float32)

    # Preencher contagens (Features 0 e 1)
    print("Agregando séries temporais...")

    # Filtros
    is_cvli = occurrences_df['tipo_evento'].astype(str).str.upper().isin(['HOMICIDIO DOLOSO', 'FEMINICÍDIO', 'ROUBO SEGUIDO DE MORTE (LATROCINIO)', 'LESAO CORPORAL SEGUIDA DE MORTE'])
    is_roubo_veiculo = occurrences_df['tipo_evento'].astype(str).str.upper().str.contains('VEÍCULO', na=False) | occurrences_df['tipo_evento'].astype(str).str.upper().str.contains('MOTO', na=False)

    occurrences_df['day_idx'] = (occurrences_df['data'] - start_date).dt.days

    valid_mask = (occurrences_df['day_idx'] >= 0) & (occurrences_df['day_idx'] < num_timesteps)
    df_valid = occurrences_df[valid_mask]

    # Agrupamento CVLI
    cvli_counts = df_valid[is_cvli[valid_mask]].groupby(['node_id', 'day_idx']).size()
    for (node, day), count in cvli_counts.items():
        features[node, day, 0] = count

    # Agrupamento Roubo Veiculo
    rv_counts = df_valid[is_roubo_veiculo[valid_mask]].groupby(['node_id', 'day_idx']).size()
    for (node, day), count in rv_counts.items():
        features[node, day, 1] = count

    # Features Estáticas (Distâncias)
    # Normalizar distâncias (Inverso ou MinMax? Vamos usar Inverso com log para "Decaimento")
    # Ou simplesmente normalizar por max. Vamos usar normalização simples por enquanto (km).
    dist_cv = nodes_gdf['dist_cv'].values / 1000.0 # km
    dist_tcp = nodes_gdf['dist_tcp'].values / 1000.0
    dist_ghost = nodes_gdf['dist_ghost'].values / 1000.0

    # Replicar no tempo
    # shape (N,) -> (N, T)
    features[:, :, 2] = np.tile(dist_cv[:, np.newaxis], (1, num_timesteps))
    features[:, :, 3] = np.tile(dist_tcp[:, np.newaxis], (1, num_timesteps))
    features[:, :, 4] = np.tile(dist_ghost[:, np.newaxis], (1, num_timesteps))

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
    
    # Calcular distâncias par a par
    # Para N~200, N^2 = 40000, rápido.
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords) # metros
    
    # Adj Geo
    mask_geo = dists <= NEIGHBOR_DIST_METERS
    adj_geo[mask_geo] = 1.0
    
    # Adj Conflict
    # Conexão se (Geo Vizinho) E (Facções Diferentes) E (Nenhuma é Neutra)
    # Se quiser tensão com neutro, remova a checagem de 'NEUTRO'.
    # O usuário disse: "Se um nó é CV e o vizinho é TCP... peso maior".
    for i in range(n):
        for j in range(n):
            if mask_geo[i, j]:
                f_i = factions[i]
                f_j = factions[j]

                # Auto-loop não é conflito
                if i == j:
                    continue

                # Definição de rivalidade direta
                is_rival = (f_i == 'CV' and f_j == 'TCP') or (f_i == 'TCP' and f_j == 'CV')

                if is_rival:
                    adj_conflict[i, j] = 1.0 # Peso de conflito
                    # Opcional: Aumentar peso geográfico também?
                    # adj_geo[i, j] = 2.0
    
    # Normalização
    # (Geralmente feito no model loader, mas garantindo diagonal)
    np.fill_diagonal(adj_geo, 1.0)
    # adj_conflict diagonal é 0
    
    return adj_geo, adj_conflict

def main():
    print("Iniciando processamento de dados (Refatorado)...")
    
    # 1. Carregar Nós (Bairros)
    nodes_gdf, nodes_proj = load_nodes_from_json(BAIRROS_FILE)
    
    # 2. Carregar Camadas de Inteligência
    layers = load_faction_layers(INTELIGENCIA_DIR)
    
    # 3. Calcular Features de Conflito nos Nós
    d_cv, d_tcp, d_ghost, factions = calculate_distances_and_faction(nodes_proj, layers)
    nodes_gdf['dist_cv'] = d_cv
    nodes_gdf['dist_tcp'] = d_tcp
    nodes_gdf['dist_ghost'] = d_ghost
    nodes_gdf['faction'] = factions
    
    print("Distribuição de Facções:")
    print(nodes_gdf['faction'].value_counts())
    
    # 4. Carregar e Atribuir Ocorrências
    occurrences_df = load_and_assign_occurrences(OCORRENCIAS_FILE, nodes_gdf)

    # 5. Construir Tensor de Features
    # Definir intervalo de datas baseado nos dados
    min_date = occurrences_df['data'].min()
    max_date = occurrences_df['data'].max()
    print(f"Intervalo de dados: {min_date.date()} a {max_date.date()}")

    node_features, dates = build_feature_tensor(nodes_gdf, occurrences_df, min_date, max_date)

    # 6. Construir Grafos
    adj_geo, adj_conflict = create_adjacency_matrices(nodes_gdf, nodes_proj)

    # 7. Salvar
    data_pack = {
        'node_features': node_features, # (N, T, F)
        'adj_geo': adj_geo,
        'adj_conflict': adj_conflict,
        'adj_matrix': adj_geo, # Compatibilidade
        'nodes_gdf': nodes_gdf,
        'dates': dates,
        'feature_names': ['CVLI', 'ROUBO_VEICULO', 'DIST_CV', 'DIST_TCP', 'DIST_GHOST']
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data_pack, f)
        
    print(f"✓ Dados processados salvos em {OUTPUT_FILE}")
    print(f"  - Shape Features: {node_features.shape}")
    print(f"  - Arestas Conflito: {np.sum(adj_conflict)}")

if __name__ == "__main__":
    main()

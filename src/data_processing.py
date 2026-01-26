import json
import os
import geopandas as gpd
import pandas as pd
import pickle
import numpy as np

# Configuração
GEOJSON_DIR = 'data/raw/inteligencia'
OCORRENCIAS_CSV = 'data/raw/View_Ocorrencias_2022_ENRIQUECIDO.csv'
OUTPUT_FILE = 'data/processed/processed_graph_data.pkl'

def load_geometries(directory):
    """Carrega todos os GeoJSONs e retorna um GeoDataFrame unificado."""
    gdf_list = []
    
    print(f"Lendo GeoJSONs de {directory}...")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Diretório de geojsons não encontrado: {directory}")

    for filename in os.listdir(directory):
        if filename.endswith('.geojson'):
            filepath = os.path.join(directory, filename)
            try:
                faction_name = os.path.splitext(filename)[0]
                gdf = gpd.read_file(filepath)
                gdf['faction'] = faction_name
                if 'name' not in gdf.columns:
                    gdf['name'] = [f"{faction_name}_{i}" for i in range(len(gdf))]
                gdf_list.append(gdf)
            except Exception as e:
                print(f"Erro ao ler {filename}: {e}")
                
    if not gdf_list:
        raise ValueError("Nenhum arquivo GeoJSON válido encontrado.")
        
    full_gdf = pd.concat(gdf_list, ignore_index=True)
    if full_gdf.crs is None:
        full_gdf.set_crs(epsg=4326, inplace=True)
        
    print(f"Total de polígonos carregados: {len(full_gdf)}")
    return full_gdf

def load_occurrences(filepath):
    """Carrega e limpa os dados de ocorrências (CSV ou JSON)."""
    print(f"Lendo ocorrências de {filepath}...")

    if filepath.lower().endswith('.json'):
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
    else:
        # CSV
        df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')

    # Normalizar nomes de colunas (strip spaces, lowercase)
    df.columns = df.columns.str.strip()

    # Encontrar e renomear latitude/longitude
    lat_col = next((c for c in df.columns if c.lower() in ('latitude', 'lat')), None)
    lon_col = next((c for c in df.columns if c.lower() in ('longitude', 'long', 'lon')), None)

    if lat_col is None or lon_col is None:
        print(f'Colunas de latitude/longitude não encontradas. Colunas disponíveis: {df.columns.tolist()}')
        return pd.DataFrame()

    df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}, inplace=True)

    # Encontrar coluna de data
    data_col = next((c for c in df.columns if c.lower() == 'data'), None)
    if data_col is None:
        data_col = next((c for c in df.columns if c.lower() in ('date', 'data_ocorrencia', 'dataocorrencia')), None)
    if data_col:
        df.rename(columns={data_col: 'data'}, inplace=True)

    # Mapear tipo (Natureza -> cvli/cvp)
    if 'tipo' not in df.columns and 'Natureza' in df.columns:
        df['tipo'] = df['Natureza']
    elif 'tipo' not in df.columns:
        df['tipo'] = 'cvp'  # padrão

    def map_tipo(val):
        """Mapeia descrição da ocorrência para tipo de crime (cvli=violento, cvp=patrimonial/droga)"""
        if not isinstance(val, str):
            return 'cvp'
        v = val.lower()
        # Crimes violentos (cvli)
        violent_keywords = ['homic', 'roubo', 'latroc', 'lesão', 'lesao', 'sequestro', 'estupro', 'homicid', 'latrocinio']
        # Crimes patrimoniais e drogas (cvp)
        drug_keywords = ['drog', 'tráfico', 'trafico', 'consumo', 'furto']
        if any(k in v for k in violent_keywords):
            return 'cvli'
        if any(k in v for k in drug_keywords):
            return 'cvp'
        return 'cvp'  # padrão

    df['tipo'] = df['tipo'].fillna('').astype(str).apply(map_tipo)

    # Limpar coordenadas
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Converter data
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    else:
        df['data'] = pd.NaT

    df = df.dropna(subset=['data'])

    gdf_occurrences = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    print(f"Total de ocorrências válidas: {len(gdf_occurrences)}")
    return gdf_occurrences

def create_adjacency_matrix(gdf, threshold_degrees=0.005):
    n = len(gdf)
    adj_matrix = np.zeros((n, n))
    print("Calculando matriz de adjacência...")
    centroids = gdf.geometry.centroid
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = centroids.iloc[i].distance(centroids.iloc[j])
            if dist < threshold_degrees:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
                
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix

def main():
    # Criar diretório de output se não existir
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)

    # Carregar ocorrências (prefere CSV se presente)
    if os.path.exists(OCORRENCIAS_CSV):
        print(f"Usando CSV: {OCORRENCIAS_CSV}")
        occurrences_gdf = load_occurrences(OCORRENCIAS_CSV)
    else:
        print(f"CSV não encontrado: {OCORRENCIAS_CSV}")
        return

    if occurrences_gdf.empty:
        print("Erro: nenhuma ocorrência válida foi carregada.")
        return

    # Tentar carregar GeoJSONs
    try:
        print(f"Tentando carregar GeoJSONs de {GEOJSON_DIR}...")
        nodes_gdf = load_geometries(GEOJSON_DIR)
    except Exception as e:
        print(f"GeoJSONs não encontrados ou erro: {e}. Gerando grade de nós a partir das ocorrências.")
        # Gerar grade de polígonos cobrindo a bounding box das ocorrências
        minx, miny, maxx, maxy = occurrences_gdf.total_bounds
        cell_size = 0.02  # graus (~2km) - ajustável
        xs = np.arange(minx, maxx + cell_size, cell_size)
        ys = np.arange(miny, maxy + cell_size, cell_size)
        polys = []
        names = []
        for i, x in enumerate(xs[:-1]):
            for j, y in enumerate(ys[:-1]):
                from shapely.geometry import box
                poly = box(x, y, x + cell_size, y + cell_size)
                polys.append(poly)
                names.append(f'grid_{i}_{j}')
        nodes_gdf = gpd.GeoDataFrame({'name': names, 'faction': names, 'geometry': polys}, crs='EPSG:4326')
        print(f"Grade gerada: {len(nodes_gdf)} polígonos")
    
    if occurrences_gdf.empty:
        print("Abortando: Sem dados de ocorrências.")
        return

    print("Realizando Spatial Join...")
    joined_gdf = gpd.sjoin(occurrences_gdf, nodes_gdf, how="inner", predicate="within")
    print(f"Ocorrências dentro das áreas monitoradas: {len(joined_gdf)}")
    
    if len(joined_gdf) == 0:
        print("Erro: nenhuma ocorrência dentro dos nós. Verifique coordenadas.")
        return
    
    min_date = joined_gdf['data'].min()
    max_date = joined_gdf['data'].max()
    print(f"Intervalo temporal: {min_date.date()} a {max_date.date()}")
    
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    nodes_gdf['node_id'] = range(len(nodes_gdf))
    
    num_nodes = len(nodes_gdf)
    num_timesteps = len(date_range)
    num_features = 2 
    
    node_features = np.zeros((num_nodes, num_timesteps, num_features))
    
    print(f"Agregando dados temporais: {num_nodes} nós, {num_timesteps} dias, {num_features} features...")
    
    # Garantir que 'tipo' existe e é lower
    if 'tipo' not in joined_gdf.columns:
        joined_gdf['tipo'] = 'cvp'
    joined_gdf['tipo'] = joined_gdf['tipo'].astype(str).str.lower()
    
    grouped = joined_gdf.groupby(['index_right', 'data', 'tipo']).size().reset_index(name='count')
    
    # Otimização vetorizada para preencher node_features
    df = grouped.copy()
    df['time_idx'] = (df['data'] - min_date).dt.days

    # Filtrar índices temporais válidos
    valid_time_mask = (df['time_idx'] >= 0) & (df['time_idx'] < num_timesteps)
    df = df[valid_time_mask]

    # Mapear tipo para índice da feature: cvli -> 0, cvp -> 1
    conditions = [
        df['tipo'].str.contains('cvli', na=False),
        df['tipo'].str.contains('cvp', na=False)
    ]
    choices = [0, 1]
    df['feature_idx'] = np.select(conditions, choices, default=-1)

    # Filtrar features válidas
    df = df[df['feature_idx'] != -1]

    # Atribuição acumulada vetorizada
    # np.add.at realiza soma unbuffered no lugar, tratando duplicatas corretamente
    np.add.at(node_features,
              (df['index_right'].values.astype(int),
               df['time_idx'].values.astype(int),
               df['feature_idx'].values.astype(int)),
              df['count'].values)
    
    print("Criando matriz de adjacência...")
    adj_matrix = create_adjacency_matrix(nodes_gdf)
    
    data_pack = {
        'node_features': node_features,
        'adj_matrix': adj_matrix,
        'nodes_gdf': nodes_gdf,
        'dates': date_range
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data_pack, f)
        
    print(f"✓ Dados processados salvos em {OUTPUT_FILE}")
    print(f"  - Nós: {num_nodes}")
    print(f"  - Timesteps (dias): {num_timesteps}")
    print(f"  - Features: {num_features}")
    print(f"  - Shape node_features: {node_features.shape}")

if __name__ == "__main__":
    main()

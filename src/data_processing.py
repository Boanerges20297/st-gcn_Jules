import json
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import pickle
import numpy as np

# Configuração
GEOJSON_DIR = 'Dados Jules/inteligencia'
OCORRENCIAS_FILE = 'Dados Jules/dados_status_ocorrencias_gerais.json'
OUTPUT_FILE = 'data/processed_graph_data.pkl'

def load_geometries(directory):
    """Carrega todos os GeoJSONs e retorna um GeoDataFrame unificado."""
    gdf_list = []
    
    print(f"Lendo GeoJSONs de {directory}...")
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
    """Carrega e limpa os dados de ocorrências."""
    print(f"Lendo ocorrências de {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    occurrences_list = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'data' in item and isinstance(item['data'], list):
                occurrences_list = item['data']
                break
    
    if not occurrences_list:
        if isinstance(data, list) and len(data) > 0:
             occurrences_list = data

    df = pd.DataFrame(occurrences_list)
    df.columns = df.columns.str.strip()

    if 'latitude' not in df.columns:
        for col in df.columns:
            if col.lower() == 'latitude':
                df.rename(columns={col: 'latitude'}, inplace=True)
            if col.lower() == 'longitude':
                df.rename(columns={col: 'longitude'}, inplace=True)

    if 'latitude' not in df.columns:
         return pd.DataFrame()

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df.dropna(subset=['data'])
    
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf_occurrences = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
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
    nodes_gdf = load_geometries(GEOJSON_DIR)
    occurrences_gdf = load_occurrences(OCORRENCIAS_FILE)
    
    if occurrences_gdf.empty:
        print("Abortando: Sem dados de ocorrências.")
        return

    print("Realizando Spatial Join...")
    joined_gdf = gpd.sjoin(occurrences_gdf, nodes_gdf, how="inner", predicate="within")
    print(f"Ocorrências dentro das áreas monitoradas: {len(joined_gdf)}")
    
    min_date = joined_gdf['data'].min()
    max_date = joined_gdf['data'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    nodes_gdf['node_id'] = range(len(nodes_gdf))
    
    num_nodes = len(nodes_gdf)
    num_timesteps = len(date_range)
    num_features = 2 
    
    node_features = np.zeros((num_nodes, num_timesteps, num_features))
    
    print("Agregando dados temporais...")
    
    if 'tipo' in joined_gdf.columns:
        joined_gdf['tipo'] = joined_gdf['tipo'].str.lower()
    else:
        joined_gdf['tipo'] = 'cvp'
    
    grouped = joined_gdf.groupby(['index_right', 'data', 'tipo']).size().reset_index(name='count')
    
    for _, row in grouped.iterrows():
        node_idx = int(row['index_right'])
        date = row['data']
        crime_type = row['tipo']
        count = row['count']
        
        time_idx = (date - min_date).days
        
        if 0 <= time_idx < num_timesteps:
            if 'cvli' in crime_type:
                node_features[node_idx, time_idx, 0] += count
            elif 'cvp' in crime_type:
                node_features[node_idx, time_idx, 1] += count
    
    adj_matrix = create_adjacency_matrix(nodes_gdf)
    
    data_pack = {
        'node_features': node_features,
        'adj_matrix': adj_matrix,
        'nodes_gdf': nodes_gdf,
        'dates': date_range
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data_pack, f)
        
    print(f"Dados processados salvos em {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

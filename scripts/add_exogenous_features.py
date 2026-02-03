"""
Script para add Features Exógenas (Prison Events + Weather)
Estende os 8 canais originais para 10 canais com features exógenas
"""

import os
import json
import numpy as np
import pandas as pd
import pickle

def load_exogenous_features():
    """Carrega eventos prisionais e dados de clima"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Eventos prisionais
    prison_events = {}
    prison_file = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia', 'presidiaria.geojson')
    if os.path.exists(prison_file):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(prison_file)
            if 'properties' in gdf.columns:
                for idx, row in gdf.iterrows():
                    props = row.get('properties', {})
                    if isinstance(props, dict):
                        date_str = props.get('data')
                        if date_str:
                            prison_events[date_str] = prison_events.get(date_str, 0) + 1
        except Exception as e:
            print(f"Erro ao carregar eventos prisionais: {e}")
    
    # Dados de clima (simulado - em produção viria de API)
    weather_data = {}
    weather_file = os.path.join(BASE_DIR, 'weather_cache', 'weather_data.json')
    if os.path.exists(weather_file):
        try:
            with open(weather_file, 'r') as f:
                weather_data = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar dados de clima: {e}")
    
    return prison_events, weather_data

def add_exogenous_features(node_features, dates, exogenous_dict=None):
    """
    Adiciona 2 canais exógenos aos 8 originais
    
    Canais adicionados:
    8: Prison Events Index (0-1)
    9: Weather Severity (0-1, normalizem chuva/temperatura)
    """
    num_nodes, num_timesteps, num_features = node_features.shape
    
    # Criar novo tensor com 10 canais
    new_features = np.zeros((num_nodes, num_timesteps, num_features + 2), dtype=np.float32)
    
    # Copiar canais originais
    new_features[:, :, :num_features] = node_features
    
    # Canal 8: Prison Events
    prison_events, weather_data = load_exogenous_features()
    
    # Mapear eventos prisionais por data
    prison_intensity = np.zeros(num_timesteps, dtype=np.float32)
    for t, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        if date_str in prison_events:
            prison_intensity[t] = min(1.0, prison_events[date_str] / 10.0)  # Normalizar
    
    # Aplicar suavização temporal (eventos prisionais têm impacto 3-7 dias depois)
    for t in range(num_timesteps):
        for lag in range(1, 8):
            if t + lag < num_timesteps:
                prison_intensity[t + lag] += prison_intensity[t] * (0.5 ** lag)
    
    prison_intensity = np.clip(prison_intensity, 0, 1)
    new_features[:, :, 8] = prison_intensity[np.newaxis, :]
    
    # Canal 9: Weather Severity
    weather_severity = np.zeros(num_timesteps, dtype=np.float32)
    for t, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        if date_str in weather_data:
            w = weather_data[date_str]
            # Combinar chuva + temperatura extrema
            rain_factor = min(1.0, w.get('precip_mm', 0) / 50.0)
            temp_factor = 1.0 if 15 <= w.get('temp_avg', 25) <= 35 else 0.5
            weather_severity[t] = (rain_factor + temp_factor) / 2
    
    new_features[:, :, 9] = weather_severity[np.newaxis, :]
    
    return new_features

def update_processed_data_with_exogenous():
    """Actualiza processed_graph_data.pkl com 10 canais"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    pkl_file = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
    
    if not os.path.exists(pkl_file):
        print(f"Erro: {pkl_file} não encontrado")
        return
    
    print("[1/2] Carregando dados originais...")
    with open(pkl_file, 'rb') as f:
        data_pack = pickle.load(f)
    
    node_features = data_pack['node_features']
    dates = data_pack['dates']
    
    print(f"     Shape original: {node_features.shape}")
    
    print("[2/2] Adicionando features exógenas...")
    new_features = add_exogenous_features(node_features, dates)
    
    print(f"     Shape novo: {new_features.shape}")
    
    # Atualizar data_pack
    data_pack['node_features'] = new_features
    data_pack['feature_names'] = [
        'CVLI', 'CVP', 'TENSION_INDEX',
        'DOW_SIN', 'DOW_COS', 'MONTH_SIN', 'MONTH_COS', 'IS_WEEKEND',
        'PRISON_EVENTS', 'WEATHER_SEVERITY'
    ]
    
    # Salvar
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_pack, f)
    
    print(f"✅ Arquivo atualizado com sucesso!")
    print(f"   Novo shape: {new_features.shape}")
    print(f"   Features: {data_pack['feature_names']}")

if __name__ == "__main__":
    update_processed_data_with_exogenous()

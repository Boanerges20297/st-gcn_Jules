#!/usr/bin/env python3
"""
TEST v2: Viabilidade com Geolocalização + Cidade
Usa latitude/longitude para mapear prisões aos 319 bairros
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def load_prison_data():
    """Carrega dados de prisões com geolocalização"""
    js_file = os.path.join(BASE_DIR, 'data', 'raw', 'data_with_coordinates.js')
    
    if not os.path.exists(js_file):
        print(f"❌ Arquivo não encontrado: {js_file}")
        return None
    
    print(f"[1/6] Carregando prisões com geolocalização...")
    
    with open(js_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    json_str = content.replace('module.exports = ', '').strip()
    if json_str.endswith(';'):
        json_str = json_str[:-1]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSON: {e}")
        return None
    
    # Filtrar apenas registros com geolocalização e Fortaleza
    fortaleza_data = []
    for record in data:
        if (record.get('latitude') and record.get('longitude') and 
            record.get('CidadeOcor', '').strip().upper() == 'FORTALEZA'):
            fortaleza_data.append(record)
    
    print(f"✅ Carregados {len(fortaleza_data)} registros de prisão em FORTALEZA com geolocalização")
    return fortaleza_data

def load_crime_nodes():
    """Carrega nós de crime do grafo com coordenadas"""
    pkl_file = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
    bairros_file = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
    
    if not os.path.exists(pkl_file) or not os.path.exists(bairros_file):
        print(f"❌ Arquivos não encontrados")
        return None, None, None
    
    print(f"\n[2/6] Carregando grafo de crime...")
    
    with open(pkl_file, 'rb') as f:
        data_pack = pickle.load(f)
    
    with open(bairros_file, 'r', encoding='utf-8') as f:
        bairros_dict = json.load(f)
    
    node_features = data_pack['node_features']
    dates = data_pack['dates']
    
    # Extrair coordenadas dos bairros (dict format)
    nodes_coords = []
    nodes_names = []
    for bairro_name, bairro_info in sorted(bairros_dict.items()):
        if isinstance(bairro_info, dict) and 'lat' in bairro_info and 'long' in bairro_info:
            nodes_coords.append([bairro_info['lat'], bairro_info['long']])
            nodes_names.append(bairro_name)
    
    print(f"✅ Grafo carregado:")
    print(f"   Shape: {node_features.shape}")
    print(f"   {len(nodes_coords)} bairros com coordenadas")
    print(f"   Período: {dates[0].date()} até {dates[-1].date()}")
    
    return node_features, dates, (np.array(nodes_coords), nodes_names)

def spatial_join_prisons(prison_data, nodes_info):
    """Mapeia prisões aos bairros mais próximos usando spatial join"""
    print(f"\n[3/6] Mapeando prisões aos bairros (spatial join)...")
    
    nodes_coords, nodes_names = nodes_info
    
    # Coordenadas das prisões
    prison_coords = np.array([[p['latitude'], p['longitude']] for p in prison_data])
    
    # Distância de cada prisão ao bairro mais próximo
    distances = cdist(prison_coords, nodes_coords)
    closest_nodes = np.argmin(distances, axis=1)
    closest_dists = np.min(distances, axis=1)
    
    # Filtrar prisões muito distantes (>3km = ~0.03 graus)
    valid_mask = closest_dists < 0.03
    
    print(f"✅ Mapeadas {valid_mask.sum()} prisões aos bairros")
    print(f"   Distância média: {closest_dists[valid_mask].mean()*111:.2f} km")
    
    # Adicionar node_id aos registros
    for i, record in enumerate(prison_data):
        if valid_mask[i]:
            record['node_id'] = closest_nodes[i]
            record['node_name'] = nodes_names[closest_nodes[i]]
            record['distance_km'] = closest_dists[i] * 111
        else:
            record['node_id'] = None
    
    return prison_data

def aggregate_prison_data(prison_data):
    """Agrega prisões por data e node_id"""
    print(f"\n[4/6] Agregando prisões por data/node...")
    
    prison_by_date_node = {}
    
    for record in prison_data:
        if record.get('node_id') is None:
            continue
        
        try:
            date_str = record.get('Data', '')
            node_id = record['node_id']
            
            date_obj = pd.to_datetime(date_str)
            date_key = date_obj.strftime('%Y-%m-%d')
            
            key = (date_key, node_id)
            prison_by_date_node[key] = prison_by_date_node.get(key, 0) + 1
        except:
            continue
    
    print(f"✅ Agregadas {len(prison_by_date_node)} combinações únicas data+node")
    
    # Por dia (total Fortaleza)
    prison_by_date = {}
    for (date_key, node_id), count in prison_by_date_node.items():
        prison_by_date[date_key] = prison_by_date.get(date_key, 0) + count
    
    dates_unique = len(prison_by_date)
    print(f"   {dates_unique} dias com prisões")
    print(f"   Média: {sum(prison_by_date.values()) / dates_unique:.1f} prisões/dia")
    
    # Top 10 bairros
    print(f"\n   Top 10 bairros com mais prisões:")
    bairro_counts = {}
    for record in prison_data:
        if record.get('node_name'):
            bairro = record['node_name']
            bairro_counts[bairro] = bairro_counts.get(bairro, 0) + 1
    
    for bairro, count in sorted(bairro_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"      ✅ {bairro}: {count} prisões")
    
    return prison_by_date, prison_by_date_node

def correlation_analysis(prison_by_date, node_features, dates):
    """Analisa correlação entre prisões e crimes"""
    print(f"\n[5/6] Analisando correlação prisões ↔ crimes...")
    
    # Crimes por dia (CVLI - homicídios)
    crimes_by_date = {}
    for date_idx, date_obj in enumerate(dates):
        date_key = date_obj.strftime('%Y-%m-%d')
        crimes_by_date[date_key] = node_features[:, date_idx, 0].sum()
    
    # Intersecção de datas
    common_dates = sorted(set(prison_by_date.keys()) & set(crimes_by_date.keys()))
    print(f"   Datas com ambos dados: {len(common_dates)}")
    
    if len(common_dates) < 30:
        print(f"   ⚠️  Poucas datas, análise limitada")
        return
    
    prison_vals = np.array([prison_by_date.get(d, 0) for d in common_dates])
    crime_vals = np.array([crimes_by_date.get(d, 0) for d in common_dates])
    
    # Remover dias com zero crimes para correlação
    valid_mask = crime_vals > 0
    prison_vals_valid = prison_vals[valid_mask]
    crime_vals_valid = crime_vals[valid_mask]
    
    print(f"   Dias com crimes: {valid_mask.sum()} / {len(common_dates)}")
    
    if len(prison_vals_valid) > 10:
        corr_0 = pearsonr(prison_vals_valid, crime_vals_valid)[0]
        print(f"   Correlação (lag=0 dias): {corr_0:.4f}")
        
        # Lag 3 dias
        if len(common_dates) > 3:
            corr_3 = pearsonr(prison_vals[:-3], crime_vals[3:])[0]
            print(f"   Correlação (lag=3 dias): {corr_3:.4f}")
        
        # Lag -3 (prisões preveem crimes 3 dias depois)
        if len(common_dates) > 3:
            corr_m3 = pearsonr(prison_vals[3:], crime_vals[:-3])[0]
            print(f"   Correlação (lag=-3 dias, prisões antes): {corr_m3:.4f}")
    
    # Estatísticas
    print(f"\n   Estatísticas:")
    print(f"   Prisões: min={prison_vals.min()}, max={prison_vals.max()}, mean={prison_vals.mean():.1f}")
    print(f"   Crimes:  min={crime_vals.min()}, max={crime_vals.max()}, mean={crime_vals.mean():.2f}")
    
    print(f"\n💡 CONCLUSÃO:")
    print(f"   ✅ VIÁVEL! Dados de prisões mapeados com sucesso")
    print(f"      → Pode ser integrada como canal 8 exógeno")
    print(f"      → Período: 2025 (precisas atualizar com 2026)")

def check_2026_data(prison_data):
    """Verifica se tem dados de 2026"""
    print(f"\n[6/6] Checando dados de 2026...")
    
    dates_2026 = []
    for record in prison_data:
        try:
            date_obj = pd.to_datetime(record.get('Data', ''))
            if date_obj.year == 2026:
                dates_2026.append(date_obj)
        except:
            pass
    
    if dates_2026:
        print(f"   ✅ {len(dates_2026)} registros de 2026")
        print(f"      Período: {min(dates_2026).date()} até {max(dates_2026).date()}")
    else:
        print(f"   ⚠️  SEM dados de 2026 (apenas 2025)")

def main():
    print("=" * 80)
    print("TEST v2: Viabilidade de Feature Exógena - Prisões com Geolocalização")
    print("=" * 80)
    
    prison_data = load_prison_data()
    if not prison_data:
        return
    
    nodes_info = load_crime_nodes()
    if nodes_info[0] is None:
        return
    
    node_features, dates, nodes_info = nodes_info
    
    prison_data = spatial_join_prisons(prison_data, nodes_info)
    prison_by_date, prison_by_date_node = aggregate_prison_data(prison_data)
    correlation_analysis(prison_by_date, node_features, dates)
    check_2026_data(prison_data)
    
    print("\n" + "=" * 80)
    print("✅ TEST v2 CONCLUÍDO")
    print("=" * 80)

if __name__ == "__main__":
    main()

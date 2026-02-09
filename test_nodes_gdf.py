#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script to check the nodes_gdf data."""

import pickle
import json
import os
import pandas as pd

BASE_DIR = 'c:\\Users\\Boanerges\\Desktop\\Projetos\\st-gcn_jules'

# Test 1: Load nodes_gdf.pkl
print("="*60)
print("TEST 1: Loading nodes_gdf.pkl")
print("="*60)
try:
    nodes_pkl = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.pkl')
    with open(nodes_pkl, 'rb') as f:
        nodes_gdf = pickle.load(f)
    
    print(f"✓ Arquivo carregado: {nodes_pkl}")
    print(f"  Tipo: {type(nodes_gdf)}")
    print(f"  Tamanho: {len(nodes_gdf)} nós")
    
    if hasattr(nodes_gdf, 'columns'):
        print(f"  Colunas: {list(nodes_gdf.columns)}")
    
    # Ver primeiras linhas
    print("\n  Primeiras 5 linhas:")
    if hasattr(nodes_gdf, 'head'):
        head = nodes_gdf.head()
        for col in ['latitude', 'longitude', 'risk_score', 'node_type', 'name', 'regiao']:
            if col in nodes_gdf.columns:
                print(f"    {col}: {head[col].tolist()[:3]}")
    
    print("\n  Estatísticas de risk_score:")
    if hasattr(nodes_gdf, 'risk_score'):
        print(f"    Min: {nodes_gdf['risk_score'].min()}")
        print(f"    Max: {nodes_gdf['risk_score'].max()}")
        print(f"    Mean: {nodes_gdf['risk_score'].mean():.2f}")
        print(f"    >= 90: {(nodes_gdf['risk_score'] >= 90).sum()} nós")
    
except Exception as e:
    print(f"✗ Erro ao carregar: {e}")

# Test 2: Load nodes_gdf.json
print("\n" + "="*60)
print("TEST 2: Loading nodes_gdf.json")
print("="*60)
try:
    nodes_json = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data', 'nodes_gdf.json')
    with open(nodes_json, 'r') as f:
        nodes_data = json.load(f)
    
    print(f"✓ Arquivo carregado: {nodes_json}")
    print(f"  Tipo: {type(nodes_data)}")
    if isinstance(nodes_data, dict):
        print(f"  Número de entradas: {len(nodes_data)}")
        # Mostrar primeira entrada
        first_key = list(nodes_data.keys())[0] if nodes_data else None
        if first_key:
            print(f"  Primeira entrada ({first_key}): {nodes_data[first_key]}")
    elif isinstance(nodes_data, list):
        print(f"  Número de elementos: {len(nodes_data)}")
        if nodes_data:
            print(f"  Primeiro elemento: {nodes_data[0]}")
            
except Exception as e:
    print(f"✗ Erro ao carregar: {e}")

# Test 3: Check bairros JSON
print("\n" + "="*60)
print("TEST 3: Checking bairros_centros_latlong.json")
print("="*60)
try:
    bairros_file = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
    with open(bairros_file, 'r', encoding='utf-8') as f:
        bairros_data = json.load(f)
    
    print(f"✓ Arquivo carregado: {bairros_file}")
    print(f"  Número de bairros: {len(bairros_data)}")
    
    # Contar tipos
    node_types = {}
    for name, info in bairros_data.items():
        ntype = info.get('node_type', 'unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1
    
    print(f"  Tipos de nós: {node_types}")
    
except Exception as e:
    print(f"✗ Erro ao carregar: {e}")

print("\n" + "="*60)

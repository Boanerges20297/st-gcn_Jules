#!/usr/bin/env python
"""
Adicionar comunidades/favelas como nós adicionais ao grafo
Extrai de COMANDO VERMELHO.geojson e outros arquivos de inteligência
"""

import json
import os
import pickle
import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTELIGENCIA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia')
pickle_path = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')

print("=" * 80)
print("🔨 ADICIONANDO COMUNIDADES/FAVELAS COMO NÓS")
print("=" * 80)

# Carrega o grafo atual
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data['nodes_gdf']
print(f"\n📍 Nós atuais: {len(nodes_gdf)}")

# Coletar todas as comunidades de todos os arquivos de inteligência
all_communities = []

for geojson_file in Path(INTELIGENCIA_DIR).glob('*.geojson'):
    print(f"\n🔍 Lendo {geojson_file.name}...")
    try:
        gdf = gpd.read_file(geojson_file)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        
        for idx, row in gdf.iterrows():
            # Extrair o nome
            name = None
            for col in ['Name', 'NAME', 'name', 'NOME', 'nome']:
                if col in row and row[col]:
                    name = str(row[col]).strip()
                    break
            
            if not name or name in ['', 'None', 'null']:
                # Tentar extrair do Description
                desc = row.get('Description', '')
                if isinstance(desc, str) and 'nome:' in desc.lower():
                    parts = desc.split('nome:')
                    if len(parts) > 1:
                        name = parts[1].split('<')[0].split('\n')[0].strip()
            
            if name and name not in ['', 'None', 'null']:
                # Certificar que não está já no grafo
                if name not in nodes_gdf['name'].values:
                    # Calcular centroide
                    try:
                        centroid = row.geometry.centroid
                        all_communities.append({
                            'name': name,
                            'latitude': centroid.y,
                            'longitude': centroid.x,
                            'regiao': 'fortaleza',  # Default para comunidades (geralmente em Fortaleza)
                            'node_type': 'comunidade',
                            'geometry': centroid,
                            'source_file': geojson_file.name
                        })
                        print(f"   ✅ {name}")
                    except Exception as e:
                        print(f"   ❌ {name}: {e}")
    except Exception as e:
        print(f"   ❌ Erro lendo {geojson_file.name}: {e}")

print(f"\n📊 Comunidades encontradas: {len(all_communities)}")

if all_communities:
    # Criar DataFrame com as comunidades
    communities_df = pd.DataFrame(all_communities)
    communities_gdf = gpd.GeoDataFrame(
        communities_df,
        geometry='geometry',
        crs="EPSG:4326"
    )
    
    # Manter o CRS do original
    if nodes_gdf.crs != communities_gdf.crs:
        communities_gdf = communities_gdf.to_crs(nodes_gdf.crs)
    
    # Adicionar as comunidades ao GeoDataFrame existente
    print(f"\n➕ Adicionando {len(communities_gdf)} comunidades ao grafo...")
    nodes_gdf_extended = pd.concat([nodes_gdf, communities_gdf], ignore_index=True)
    
    print(f"\n✅ Novo total de nós: {len(nodes_gdf_extended)}")
    print(f"   - Bairros: {len(nodes_gdf_extended[nodes_gdf_extended['node_type'] == 'bairro'])}")
    print(f"   - Cidades: {len(nodes_gdf_extended[nodes_gdf_extended['node_type'] == 'cidade'])}")
    print(f"   - Comunidades: {len(nodes_gdf_extended[nodes_gdf_extended['node_type'] == 'comunidade'])}")
    
    # Atualizar o pickle
    print(f"\n💾 Salvando dados atualizados em {pickle_path}...")
    data['nodes_gdf'] = nodes_gdf_extended
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ Salvo com sucesso!")
    
    # Listar as comunidades adicionadas
    print(f"\n📋 Comunidades adicionadas:")
    communities_only = nodes_gdf_extended[nodes_gdf_extended['node_type'] == 'comunidade']
    for idx, row in communities_only.iterrows():
        print(f"   - {row['name']} (de {row.get('source_file', 'desconhecido')})")
else:
    print("\n⚠️ Nenhuma comunidade encontrada para adicionar. Verificar argumentos de coluna.")

print("\n" + "=" * 80)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download da Malha Viária usando OSMnx.
Este script extrai os grafos rodoviários (apenas para tráfego viaxe / drive)
de Fortaleza, da RMF e (opcionalmente) de municípios do interior.
O grafo é convertido para GeoJSON para utilização no ST-GCN/Dashboard e para
melhorar a granularidade do modelo.
"""

import os
import sys
import geopandas as gpd

# Adicionar PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import osmnx as ox

# Configurações de Download
# Apenas Fortaleza e principais focos da RMF para não estourar RAM primeiramente
PLACES_TO_DOWNLOAD = [
    "Fortaleza, Ceará, Brazil",
    "Caucaia, Ceará, Brazil",
    "Maracanaú, Ceará, Brazil",
    #"Sobral, Ceará, Brazil", # Exemplo de expansão para interior
    #"Juazeiro do Norte, Ceará, Brazil"
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'static')

def download_road_network():
    print(f"[{__file__}] Iniciando o download da malha viária...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configurações otimizadas do OSMnx
    ox.settings.log_console = True
    ox.settings.use_cache = True
    
    graphs = []
    
    # Baixar Grafo de cada localidade
    for place in PLACES_TO_DOWNLOAD:
        print(f"Baixando: {place}")
        try:
            # network_type='drive' garante apenas ruas transitáveis por veículos
            G = ox.graph_from_place(place, network_type='drive', simplify=True)
            graphs.append(G)
        except Exception as e:
            print(f"Erro ao baixar {place}: {e}")
            
    if not graphs:
        print("Nenhum grafo foi baixado. Cancelando script.")
        return

    print(f"Realizando o merge de {len(graphs)} grafos...")
    import networkx as nx
    G_merged = nx.compose_all(graphs)
    print("Grafos mesclados com sucesso.")
    
    print("Convertendo grafos para GeoDataFrames...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_merged)
    
    # Limpando MultiIndex e colunas que contenham listas (GeoJSON n suporta arrays mistos)
    edges_gdf = edges_gdf.reset_index()
    
    print("Tratando colunas para exportação GeoJSON...")
    for col in edges_gdf.columns:
        if col != 'geometry':
            # Se for lista, converte para string separada por vírgula
            edges_gdf[col] = edges_gdf[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
                if not pd.isna(x) else ""
            )

    edges_path = os.path.join(OUTPUT_DIR, "malha_viaria_edges.geojson")
    graphml_path = os.path.join(OUTPUT_DIR, "malha_viaria.graphml")
    
    print(f"Salvando Grafo topológico (GraphML) para inferência ST-GCN em: {graphml_path}")
    ox.save_graphml(G_merged, graphml_path)
    
    print(f"Salvando GeoJSON em: {edges_path}")
    edges_gdf.to_file(edges_path, driver="GeoJSON")
    
    print(f"✓ Concluído! Total de {len(nodes_gdf)} nós e {len(edges_gdf)} ruas exportados.")

if __name__ == "__main__":
    import pandas as pd
    download_road_network()

import os, sys, json, warnings, unicodedata
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon, MultiPolygon

# Configurações de exibição
warnings.filterwarnings("ignore")
ox.settings.use_cache = True
ox.settings.log_console = False

# Caminhos base
BASE_PATH = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
DATA_RAW = os.path.join(BASE_PATH, "data", "raw")
DATA_STATIC = os.path.join(BASE_PATH, "data", "static")
NETWORK_CACHE = os.path.join(BASE_PATH, "data", "network_cache")
OUT_SENTINELA = os.path.join(BASE_PATH, "tests", "Sentinela")

if not os.path.exists(NETWORK_CACHE):
    os.makedirs(NETWORK_CACHE)

def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def identify_active_neighborhoods(days_window=60, ratio_threshold=20):
    """
    Identifica bairros com atividade de CVLI > 1 e com perfil de periculosidade pura.
    Remove bairros de perfil estritamente patrimonial (ex: Amadeu Furtado).
    """
    csv_path = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
    if not os.path.exists(csv_path):
        return []

    print(f"Lendo crimes para filtragem de pureza (Janela: {days_window} dias)...")
    df = pd.read_csv(csv_path, low_memory=False)
    df["tipo"] = df["tipo"].astype(str).str.lower()
    
    # 1. Calcular volume por categoria no periodo
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data", "bairro"])
    df["bairro_norm"] = df["bairro"].apply(norm)
    
    max_date = df["data"].max()
    cutoff_date = max_date - timedelta(days=days_window)
    recent = df[df["data"] >= cutoff_date]
    
    # Agrupar por tipo e bairro
    stats = recent.groupby(["bairro_norm", "tipo"]).size().unstack(fill_value=0)
    
    if "cvli" not in stats.columns: return []
    if "cvp" not in stats.columns: stats["cvp"] = 0
    
    # 2. Aplicar Filtros: CVLI >= 1 E Ratio CVP/CVLI < threshold
    stats["cvli_ratio"] = stats["cvp"] / (stats["cvli"] + 0.1)
    
    # Filtrar: Ativos (pelo menos 1 CVLI) e Puros (baixa densidade de CVP relativo)
    active_mask = (stats["cvli"] >= 1) & (stats["cvli_ratio"] <= ratio_threshold)
    active = stats[active_mask].index.tolist()
    
    ignored = stats[(stats["cvli"] >= 1) & (stats["cvli_ratio"] > ratio_threshold)].index.tolist()
    if ignored:
        print(f"  Aviso: {len(ignored)} bairros patrimoniais ignorados (ex: {ignored[:3]}).")
        
    print(f"Ok: {len(active)} bairros/cidades criticos identificados (Pureza CVLI Garantida).")
    return active

def get_neighborhood_data_sources(neighborhood_names):
    """
    Mapeia locais para Coordenadas (Fortaleza) ou Busca Nominal (Interior).
    ABANDONA O MODELO AIS.
    """
    coords_path = os.path.join(DATA_STATIC, "fortaleza_bairros_coords.json")
    bairro_map = {}
    
    # 1. Carregar Coordenadas Fortaleza (Fonte Definida pelo Usuario)
    if os.path.exists(coords_path):
        with open(coords_path, "r", encoding="utf-8") as f:
            raw_coords = json.load(f)
            for b_name, coords in raw_coords.items():
                bairro_map[norm(b_name)] = {"type": "point", "data": coords}
                
    mapped_sources = {}
    for name in neighborhood_names:
        if name in bairro_map:
            mapped_sources[name] = bairro_map[name]
        else:
            # 2. Busca Nominal OSMNX para Municípios/RMF fora do JSON
            mapped_sources[name] = {"type": "place", "data": f"{name}, Ceara, Brazil"}
            
    return mapped_sources

def extract_and_zone_networks(active_sources):
    """
    Extrai malha viária e define zonas de 500m.
    """
    zone_data = []
    
    for name, source in active_sources.items():
        cache_file = os.path.join(NETWORK_CACHE, f"graph_{name}.graphml")
        print(f"Processando malha viaria: {name} ({source['type']})...")
        
        try:
            G = None
            if os.path.exists(cache_file):
                G = ox.load_graphml(cache_file)
            else:
                if source['type'] == "point":
                    lat, lng = source['data']
                    G = ox.graph_from_point((lat, lng), dist=1000, network_type='drive')
                else:
                    try:
                        # Tenta extração por nome de lugar
                        G = ox.graph_from_place(source['data'], network_type='drive')
                    except Exception as e:
                        print(f"  Aviso: place falhou para {name}, tentando geocode centróide...")
                        try:
                            # Fallback: Geocodifica o ponto central e extrai raio de 1km
                            lat, lng = ox.geocode(source['data'])
                            G = ox.graph_from_point((lat, lng), dist=1000, network_type='drive')
                        except:
                            print(f"  Erro: Nao foi possivel localizar {name}.")
                
                if G: ox.save_graphml(G, cache_file)
            
            if G:
                nodes = ox.graph_to_gdfs(G, edges=False)
                # Filtro de sanidade
                nodes = nodes[nodes.geometry.notnull()]
                if not nodes.empty:
                    nodes['degree'] = dict(G.degree()).values()
                    top_intersections = nodes.sort_values(by='degree', ascending=False).head(5)
                    
                    for i, (idx, node) in enumerate(top_intersections.iterrows()):
                        zone_center = node['geometry']
                        zone_data.append({
                            "bairro": name,
                            "zone_id": f"{name}_Z{i+1}",
                            "lat": zone_center.y,
                            "lng": zone_center.x,
                            "geometry": zone_center.buffer(0.0045) # ~500m
                        })
        except Exception as e:
            print(f"Erro ao processar {name}: {e}")
            
    return gpd.GeoDataFrame(zone_data, crs="EPSG:4324") if zone_data else None

def main():
    print("[Sentinela V4] Iniciando Cobertura Total (Point-Radius + Fallback)...")
    active_names = identify_active_neighborhoods()
    if not active_names: return
    
    active_sources = get_neighborhood_data_sources(active_names)
    print(f"Fontes de dados encontradas para {len(active_sources)} bairros (de {len(active_names)} ativos).")
    
    zones_gdf = extract_and_zone_networks(active_sources)
    
    if zones_gdf is not None:
        out_zones = os.path.join(OUT_SENTINELA, "sentinela_v4_zones.geojson")
        zones_gdf.to_file(out_zones, driver='GeoJSON')
        print(f"Sucesso: {len(zones_gdf)} Zonas de 500m geradas em {len(zones_gdf['bairro'].unique())} bairros.")
    else:
        print("Erro: Nenhuma zona gerada.")

if __name__ == "__main__":
    main()

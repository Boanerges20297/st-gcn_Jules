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
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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

def identify_active_neighborhoods(days_window=365, ratio_threshold=30):
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
    
    # LIMPEZA TÁTICA FINAL: Remover regiões administrativas do IBGE que travam o OSMNX
    active = [a for a in active if "REGIAO GEOGRAFICA" not in a and "IMEDIATA" not in a and len(a) < 30]
    
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
    Extrai malha viária e define zonas de 500m com Modo Verboso.
    """
    zone_data = []
    total = len(active_sources)
    
    # Configurar timeout do OSMNX para evitar travamentos infinitos
    import osmnx as ox
    ox.settings.timeout = 30
    
    for i, (name, source) in enumerate(active_sources.items(), 1):
        safe_name = name.replace("/", "_").replace("\\", "_").strip()
        cache_file = os.path.join(NETWORK_CACHE, f"graph_{safe_name}.graphml")
        print(f"[{i}/{total}] Processando: {name} ({source['type']})...", flush=True)
        
        try:
            G = None
            if os.path.exists(cache_file):
                # print(f"  (Cache encontrado)")
                G = ox.load_graphml(cache_file)
            else:
                if source['type'] == "point":
                    lat, lng = source['data']
                    G = ox.graph_from_point((lat, lng), dist=1000, network_type='drive')
                else:
                    try:
                        G = ox.graph_from_place(source['data'], network_type='drive')
                    except:
                        # Fallback agressivo por geocodificação
                        try:
                            lat, lng = ox.geocode(source['data'])
                            G = ox.graph_from_point((lat, lng), dist=1000, network_type='drive')
                        except: pass
                
                if G: ox.save_graphml(G, cache_file)
            
            if G:
                nodes = ox.graph_to_gdfs(G, edges=False)
                nodes = nodes[nodes.geometry.notnull()]
                if not nodes.empty:
                    nodes['degree'] = dict(G.degree()).values()
                    # Selecionar as 5 interseções mais críticas de cada zona
                    top_intersections = nodes.sort_values(by='degree', ascending=False).head(5)
                    
                    for z_idx, (idx, node) in enumerate(top_intersections.iterrows()):
                        zone_center = node['geometry']
                        zone_data.append({
                            "bairro": name,
                            "zone_id": f"{name}_Z{z_idx+1}",
                            "lat": zone_center.y,
                            "lng": zone_center.x,
                            "geometry": zone_center.buffer(0.0045) # ~500m
                        })
        except Exception as e:
            print(f"  AVISO: Falha ao processar {name} (ignorado): {e}")
            
    return gpd.GeoDataFrame(zone_data, crs="EPSG:4324") if zone_data else None

def main():
    print("[Sentinela V4] Iniciando Cobertura Total (Resiliente)...")
    # Janela de 1 ano para pegar RMF e polos do Interior
    active_names = identify_active_neighborhoods(days_window=365)
    if not active_names: 
        print("Nenhum bairro ativo encontrado.")
        return
    
    # Garantir que cidades polo da RMF/Interior estejam sempre presentes para mapeamento de malha
    POLOS = ['CAUCAIA', 'MARACANAU', 'MARANGUAPE', 'HORIZONTE', 'PACATUBA', 'SOBRAL', 'JUAZEIRO DO NORTE', 'AQUIRAZ', 'ITAITINGA', 'QUIXADA', 'CRATO', 'IGUATU']
    for p in POLOS:
        if p not in active_names:
            active_names.append(p)

    active_sources = get_neighborhood_data_sources(active_names)
    print(f"Malha: {len(active_sources)} locais serao processados.")
    
    zones_gdf = extract_and_zone_networks(active_sources)
    
    if zones_gdf is not None:
        out_zones = os.path.join(OUT_SENTINELA, "sentinela_v4_zones.geojson")
        zones_gdf.to_file(out_zones, driver='GeoJSON')
        print(f"SUCESSO: {len(zones_gdf)} Zonas geradas em {len(zones_gdf['bairro'].unique())} locais.")
    else:
        print("Erro: Nenhuma zona gerada.")

if __name__ == "__main__":
    main()

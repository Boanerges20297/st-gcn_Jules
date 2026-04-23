import os, sys, json, warnings, unicodedata
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta

# Configurações
warnings.filterwarnings("ignore")

# Caminhos base
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(BASE_PATH, "data", "raw")
OUT_SENTINELA = os.path.join(BASE_PATH, "tests", "Sentinela")

# Fórmulas de Pesos (Conforme diretrizes táticas)
PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO': 15.0, 'PORTE ILEGAL ART 14': 12.0,
    'TRAFICO DE DROGAS': 8.0, 'APREENSAO DE DROGAS': 6.0,
    'MANDADO DE PRISAO': 4.0,
    'MANDADO EM ABERTO': 3.5, 'MANDADO DE PRISAO EM ABERTO': 3.5,
    'ABANDONO DE MATERIAL': 1.5, 'NAO INFORMADA': 0.5,
}

def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def process_intentionality():
    print("[Sentinela V4] Iniciando Motor de Intencionalidade...")
    
    # 1. Carregar Zonas
    zones_path = os.path.join(OUT_SENTINELA, "sentinela_v4_zones.geojson")
    if not os.path.exists(zones_path):
        print("Erro: Zonas de 500m nao encontradas. Rode o preprocess_sentinela_v4.py primeiro.")
        return
    zones_gdf = gpd.read_file(zones_path)
    
    # 2. Carregar Crimes (ENRIQUECIDO) - Janela de 60 dias
    occ_path = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
    df_occ = pd.read_csv(occ_path, low_memory=False)
    df_occ["data"] = pd.to_datetime(df_occ["data"], errors="coerce")
    
    max_date = df_occ["data"].max()
    cutoff_60 = max_date - timedelta(days=60)
    
    df_cvli = df_occ[(df_occ["tipo"].str.lower() == "cvli") & (df_occ["data"] >= cutoff_60)].copy()
    df_cvli = df_cvli.dropna(subset=["latitude", "longitude"])
    cvli_gdf = gpd.GeoDataFrame(
        df_cvli, geometry=gpd.points_from_xy(df_cvli.longitude, df_cvli.latitude), crs="EPSG:4326"
    )
    
    # 3. Carregar Apreensões (Tropa) - Janela de 60 dias
    tropa_path = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
    df_tropa = pd.read_csv(tropa_path, low_memory=False)
    df_tropa["data"] = pd.to_datetime(df_tropa["data"], errors="coerce")
    df_tropa = df_tropa[df_tropa["data"] >= cutoff_60].copy()
    # Assumindo que a base de tropa tenha bairro/cidade, mas não necessariamente lat/long exata em todas
    # Se não houver lat/long na tropa, usaremos o centroide do bairro para aproximar o intel do bairro para as zonas.
    # Mas se houver lat/long (ex: em algumas versões existe), usaremos.
    # Para este projeto, vamos usar o cruzamento por BAIRRO se lat/long for nulo.
    
    df_tropa["score_intel"] = (
        df_tropa["qtd_armas"] * 15.0 + 
        np.log1p(df_tropa["qtd_drogas"].fillna(0)) * 4.0 +
        df_tropa["qtd_drogas_itens"] * 2.0 +
        df_tropa["qtd_veiculos_apreendidos"] * 3.0
    )
    
    # Adicionar peso por natureza
    df_tropa["peso_nat"] = df_tropa["natureza"].str.upper().str.strip().map(lambda x: PESO_NATUREZA.get(x, 1.0))
    df_tropa["score_intel"] += df_tropa["peso_nat"]
    
    # 4. Spatial Join: Crimes -> Zonas
    cvli_in_zones = gpd.sjoin(cvli_gdf.to_crs(zones_gdf.crs), zones_gdf, how="inner", predicate="within")
    cvli_counts = cvli_in_zones.groupby("zone_id").size().reset_index(name="cvli_total")
    
    # 5. Mapear Intel para Zonas (via Bairro como fallback se sjoin falhar por falta de coords na tropa)
    # Como a base de tropa atual (conforme vimos no head) nao mostra lat/long, usaremos distribuicao proporcional por bairro
    intel_bairro = df_tropa.groupby(df_tropa["bairro"].apply(norm))["score_intel"].sum().reset_index()
    
    # 6. Consolidar no zones_gdf
    zones_gdf = zones_gdf.merge(cvli_counts, on="zone_id", how="left").fillna(0)
    zones_gdf["bairro_norm"] = zones_gdf["bairro"].apply(norm)
    zones_gdf = zones_gdf.merge(intel_bairro, left_on="bairro_norm", right_on="bairro", how="left", suffixes=("", "_b")).fillna(0)
    
    # 7. Calculo de Intencionalidade e GAP
    # GAP: Onde ha CVLI alto e Intel Baixa (Vacuo Policial)
    # Intencionalidade: Pressao criminal (Intel + CVLI)
    zones_gdf["gap_index"] = zones_gdf["cvli_total"] / (zones_gdf["score_intel"] + 0.1)
    zones_gdf["intent_index"] = (zones_gdf["cvli_total"] * 2.0 + zones_gdf["score_intel"]) / 3.0
    
    # 8. Integrar Facções (Micronodos)
    faccoes_path = os.path.join(DATA_RAW, "inteligencia", "micronodos_faccoes_2026.csv")
    if os.path.exists(faccoes_path):
        df_f = pd.read_csv(faccoes_path, low_memory=False)
        df_f["bairro_f"] = df_f["area_official"].apply(norm) if "area_official" in df_f.columns else df_f["area_oficial"].apply(norm)
        faccies_bairro = df_f.groupby("bairro_f")["faction"].first().reset_index()
        zones_gdf = zones_gdf.merge(faccies_bairro, left_on="bairro_norm", right_on="bairro_f", how="left").fillna({"faction": "NEUTRO"})
        
    # Salvar Resultado
    out_path = os.path.join(OUT_SENTINELA, "sentinela_v4_intelligence.csv")
    zones_gdf.drop(columns="geometry").to_csv(out_path, index=False)
    
    print(f"Sucesso: Motor de Intencionalidade concluido. Salvo em {out_path}")
    print(zones_gdf[["zone_id", "cvli_total", "score_intel", "gap_index", "faction"]].head(10))

if __name__ == "__main__":
    process_intentionality()

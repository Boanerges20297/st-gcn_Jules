import pandas as pd
import geopandas as gpd
import json, os, warnings
from datetime import datetime, timedelta

# Configurações
warnings.filterwarnings("ignore")

# Caminhos
BASE_PATH = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
DATA_RAW = os.path.join(BASE_PATH, "data", "raw")
OUT_SENTINELA = os.path.join(BASE_PATH, "tests", "Sentinela")

def run_shadow_validation():
    print("[Sentinela V4] Iniciando Validacao Sombra (Shadow Validation)...")
    
    # 1. Carregar Zonas
    zones_path = os.path.join(OUT_SENTINELA, "sentinela_v4_zones.geojson")
    if not os.path.exists(zones_path):
        print("Erro: Zonas V4 nao encontradas.")
        return
    zones_gdf = gpd.read_file(zones_path)
    
    # 2. Carregar Crimes Reais (Ultimos 14 dias do dataset para validar o horizonte)
    occ_path = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
    df_occ = pd.read_csv(occ_path, low_memory=False)
    df_occ["data"] = pd.to_datetime(df_occ["data"], errors="coerce")
    
    max_date = df_occ["data"].max()
    cutoff_val = max_date - timedelta(days=14)
    
    real_cvli = df_occ[(df_occ["tipo"].str.lower() == "cvli") & (df_occ["data"] >= cutoff_val)].copy()
    real_cvli = real_cvli.dropna(subset=["latitude", "longitude"])
    
    real_gdf = gpd.GeoDataFrame(
        real_cvli, geometry=gpd.points_from_xy(real_cvli.longitude, real_cvli.latitude), crs="EPSG:4326"
    )
    
    # 3. Spatial Join: Crimes Reais -> Zonas V4
    hits = gpd.sjoin(real_gdf.to_crs(zones_gdf.crs), zones_gdf, how="inner", predicate="within")
    
    # 4. Calcular Metricas
    total_crimes = len(real_gdf)
    crimes_capturados = len(hits)
    taxa_cobertura = (crimes_capturados / total_crimes * 100) if total_crimes > 0 else 0
    
    # 5. Gerar Relatorio
    report = [
        "====================================================",
        "   RELATORIO DE VALIDACAO SOMBRA - SENTINELA V4",
        "====================================================",
        f"Data do Relatorio: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"Periodo de Analise: {cutoff_val.date()} ate {max_date.date()} (14 dias)",
        f"Total de CVLIs reais no periodo: {total_crimes}",
        f"CVLIs capturados pelas Zonas V4: {crimes_capturados}",
        f"Taxa de Cobertura Espacial: {taxa_cobertura:.2f}%",
        "\nTOP ZONAS COM ACERTOS (HITS):",
    ]
    
    if not hits.empty:
        top_hits = hits.groupby("zone_id").size().sort_values(ascending=False).head(10)
        for zid, count in top_hits.items():
            report.append(f" - {zid}: {count} crimes")
    else:
        report.append(" - Nenhuma zona capturou crimes neste periodo.")
        
    report_text = "\n".join(report)
    print(report_text)
    
    with open(os.path.join(OUT_SENTINELA, "shadow_validation_report_v4.txt"), "w") as f:
        f.write(report_text)
    
    print(f"\nSucesso: Relatorio salvo em {os.path.join(OUT_SENTINELA, 'shadow_validation_report_v4.txt')}")

if __name__ == "__main__":
    run_shadow_validation()

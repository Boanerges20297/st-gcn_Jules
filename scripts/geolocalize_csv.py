import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys

def geolocalize_csv(input_csv, output_csv=None):
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_geolocalizado.csv')
    
    print(f"Lendo arquivo: {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return

    # Identificar colunas de lat/long
    lat_cols = [c for c in df.columns if c.lower() in ['latitude', 'lat', 'y']]
    long_cols = [c for c in df.columns if c.lower() in ['longitude', 'long', 'lng', 'x']]

    if not lat_cols or not long_cols:
        print("Colunas de latitude/longitude não encontradas!")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        return

    lat_col = lat_cols[0]
    long_col = long_cols[0]
    print(f"Usando colunas: {lat_col}, {long_col}")

    # Remover linhas sem coordenadas
    original_len = len(df)
    df = df.dropna(subset=[lat_col, long_col])
    if len(df) < original_len:
        print(f"Removidas {original_len - len(df)} linhas com coordenadas nulas.")

    # Converter para GeoDataFrame
    geometry = [Point(xy) for xy in zip(df[long_col], df[lat_col])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Caminhos para os arquivos GeoJSON (ajustados para a estrutura do projeto)
    base_path = r'c:\Users\Boanerges\Desktop\Projetos\Report Preview\data\static'
    neighborhoods_path = os.path.join(base_path, 'nodes_polygons.geojson')
    municipalities_path = os.path.join(base_path, 'municipios_ceara_ibge.geojson')

    # Carregar Bairros (Fortaleza)
    print("Carregando polígonos de bairros...")
    gdf_bairros = gpd.read_file(neighborhoods_path)
    if gdf_bairros.crs != "EPSG:4326":
        gdf_bairros = gdf_bairros.to_crs("EPSG:4326")

    # Join Espacial com Bairros
    print("Realizando join espacial com bairros...")
    # nodes_polygons.geojson usa 'NOME' para o bairro
    joined = gpd.sjoin(gdf_points, gdf_bairros[['NOME', 'geometry']], how='left', predicate='within')
    joined = joined.rename(columns={'NOME': 'bairro_geo'})

    # Carregar Municípios (Ceará) como fallback
    print("Carregando polígonos de municípios...")
    gdf_mun = gpd.read_file(municipalities_path)
    if gdf_mun.crs != "EPSG:4326":
        gdf_mun = gdf_mun.to_crs("EPSG:4326")

    # Join Espacial com Municípios para pontos sem bairro
    print("Realizando join espacial com municípios...")
    unmatched = joined[joined['bairro_geo'].isna()].copy()
    if not unmatched.empty:
        # municipios_ceara_ibge.geojson usa 'NM_MUN'
        mun_joined = gpd.sjoin(unmatched.drop(columns=['bairro_geo', 'index_right']), 
                               gdf_mun[['NM_MUN', 'geometry']], how='left', predicate='within')
        joined.loc[joined['bairro_geo'].isna(), 'bairro_geo'] = mun_joined['NM_MUN']

    # Finalizar DataFrame
    # Se já existir uma coluna 'bairro', podemos decidir se sobrescrevemos ou preenchemos apenas nulos
    if 'bairro' in joined.columns:
        joined['bairro'] = joined['bairro'].fillna(joined['bairro_geo'])
    else:
        joined['bairro'] = joined['bairro_geo']

    # Limpeza
    final_df = pd.DataFrame(joined.drop(columns=['geometry', 'index_right', 'bairro_geo']))

    print(f"Salvando resultado em: {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print("Concluído com sucesso!")

if __name__ == "__main__":
    input_path = r'c:\Users\Boanerges\Documents\dados_status.csv'
    geolocalize_csv(input_path)

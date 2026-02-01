import os
import geopandas as gpd
import pandas as pd

DATA_DIR = 'data/raw'

EXPECTED_FILES = [
    'ceara_municipios.geojson',
    'fortaleza_bairros.geojson',
    'ceara_interior.geojson'
]

EXISTING_FILES = [
    'limites_ceara_ibge_completo.geojson',
    'limites_ceara_ibge_simples.geojson',
    'limites_ceara.geojson',
    'limites_ceara_ibge_linhas.geojson'
]

def check_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ MISSING: {filename}")
        return False

    print(f"✅ FOUND: {filename}")
    try:
        gdf = gpd.read_file(path)
        print(f"   - Features: {len(gdf)}")
        print(f"   - Columns: {list(gdf.columns)}")
        if not gdf.empty:
            print(f"   - Sample Name: {gdf.iloc[0].get('name', 'N/A')}")
            print(f"   - Geometry Type: {gdf.iloc[0].geometry.geom_type}")
    except Exception as e:
        print(f"   - Error reading file: {e}")
    return True

def main():
    print("--- Checking Expected Polygon Files ---")
    found_any = False
    for f in EXPECTED_FILES:
        if check_file(f):
            found_any = True

    print("\n--- Inspecting Existing 'Limites' Files ---")
    for f in EXISTING_FILES:
        check_file(f)

    if not found_any:
        print("\nSUMMARY: None of the specific municipality/neighborhood polygon files were found.")
        print("Please upload:")
        print("  - data/raw/ceara_municipios.geojson (for Municipalities)")
        print("  - data/raw/fortaleza_bairros.geojson (for Fortaleza Neighborhoods)")

if __name__ == "__main__":
    main()

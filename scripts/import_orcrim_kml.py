import pandas as pd
import geopandas as gpd
import os
import unicodedata
import re
import json
import zipfile
import shutil
from xml.etree import ElementTree as ET
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt

# --- CONFIGURACOES ---
BASE_DIR = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview'
INTEL_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia')
DICT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')
DOWNLOADS_DIR = r'C:\Users\Boanerges\Downloads'

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * 6371

def normalize_text(text):
    if not text: return ""
    n = "".join(c for c in unicodedata.normalize('NFKD', str(text)) if unicodedata.category(c) != 'Mn')
    n = n.upper().strip()
    n = re.sub(r'\s*-\s*AIS.*$', '', n)
    n = re.sub(r'\s*-\s*(CV|PCC|GDE|TCP|MASSA|OKAIDA).*', '', n)
    return n

def import_kml():
    # 1. Localizar arquivo ORCRIM (KML ou KMZ)
    files = [f for f in os.listdir(DOWNLOADS_DIR) if 'ORCRIM' in f.upper()]
    if not files:
        print("❌ Nenhum arquivo ORCRIM encontrado em Downloads.")
        return
    
    latest_file = os.path.join(DOWNLOADS_DIR, sorted(files)[-1])
    print(f"📂 Processando: {latest_file}")
    
    kml_working_path = os.path.join(INTEL_DIR, 'current_orcrim.kml')
    
    if latest_file.lower().endswith('.kmz'):
        with zipfile.ZipFile(latest_file, 'r') as zip_ref:
            zip_ref.extractall(INTEL_DIR)
            doc_kml = os.path.join(INTEL_DIR, 'doc.kml')
            if os.path.exists(doc_kml):
                if os.path.exists(kml_working_path): os.remove(kml_working_path)
                os.rename(doc_kml, kml_working_path)
    else:
        shutil.copy(latest_file, kml_working_path)

    # 2. Carregar Dicionário Oficial
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        official_dict = json.load(f)
    official_names = {normalize_text(name): name for name in official_dict.keys()}
    centers = [{'name': name, 'lat': coords['lat'], 'long': coords['long']} for name, coords in official_dict.items()]

    # 3. Parse KML
    tree = ET.parse(kml_working_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    micronodos = []
    for folder in root.findall('.//kml:Folder', ns):
        fname = folder.find('kml:name', ns).text.upper() if folder.find('kml:name', ns) is not None else ""
        faction = "NEUTRO"
        if "COMANDO VERMELHO" in fname or " CV " in fname: faction = "CV"
        elif "TCP" in fname or "GDE" in fname: faction = "TCP/GDE"
        elif "PCC" in fname: faction = "PCC"
        elif "MASSA" in fname: faction = "MASSA"
        elif "OKAIDA" in fname: faction = "OKAIDA"
        elif "DISPUTA" in fname: faction = "DISPUTA"
        
        if faction == "NEUTRO": continue
        
        for pm in folder.findall('.//kml:Placemark', ns):
            name_raw = pm.find('kml:name', ns).text if pm.find('kml:name', ns) is not None else "S/N"
            coords_elem = pm.find('.//kml:coordinates', ns)
            lat, lon = None, None
            if coords_elem is not None and coords_elem.text:
                try:
                    c = coords_elem.text.strip().split()[0].split(',')
                    lon, lat = float(c[0]), float(c[1])
                except: pass
            
            # Match
            area_id = "DESCONHECIDO"
            norm_name = normalize_text(name_raw)
            if norm_name in official_names:
                area_id = official_names[norm_name]
            elif lat and lon:
                min_dist = float('inf')
                for c in centers:
                    d = haversine(lon, lat, c['long'], c['lat'])
                    if d < min_dist:
                        min_dist = d; area_id = c['name']
            
            micronodos.append({'micronodo': name_raw, 'area_oficial': area_id, 'faction': faction, 'lat': lat, 'long': lon})

    df_micro = pd.DataFrame(micronodos)
    
    # 4. Agregacao
    counts = df_micro.groupby(['area_oficial', 'faction']).size().reset_index(name='n')
    df_agregado = counts.sort_values('n', ascending=False).drop_duplicates('area_oficial')
    
    final_rows = []
    for off in official_dict.keys():
        row = df_agregado[df_agregado['area_oficial'] == off]
        f = row.iloc[0]['faction'] if not row.empty else 'NEUTRO'
        g = 0.85 if not row.empty else 0.0
        final_rows.append({'local': off, 'faccao_predominante': f, 'grau_dominio': g})
    
    df_final = pd.DataFrame(final_rows)

    # 5. Salvar e Limpar
    df_micro.to_csv(os.path.join(INTEL_DIR, 'micronodos_faccoes_2026.csv'), index=False)
    df_final.to_csv(os.path.join(INTEL_DIR, 'bairros_faccoes_2026.csv'), index=False)
    df_final.to_csv(os.path.join(BASE_DIR, 'data', 'raw', 'inteligencia_faccoes.csv'), index=False)
    
    # Gerar GeoJSON para o mapa
    gdf = gpd.GeoDataFrame(df_micro.dropna(subset=['lat', 'long']), 
                           geometry=[Point(xy) for xy in zip(df_micro.dropna(subset=['lat', 'long'])['long'], df_micro.dropna(subset=['lat', 'long'])['lat'])], 
                           crs='EPSG:4326')
    with open(os.path.join(INTEL_DIR, 'micronodos_faccoes_2026.geojson'), 'w') as f:
        f.write(gdf.to_json())

    print("✅ Inteligencia Territorial Atualizada com sucesso.")

if __name__ == "__main__":
    import_kml()

import pandas as pd
import os
import unicodedata
import re
import json
from xml.etree import ElementTree as ET
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """Calcula a distancia entre dois pontos em km."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

def normalize_text(text):
    if not text: return ""
    n = "".join(c for c in unicodedata.normalize('NFKD', str(text)) if unicodedata.category(c) != 'Mn')
    n = n.upper().strip()
    n = re.sub(r'\s*-\s*AIS.*$', '', n)
    n = re.sub(r'\s*-\s*(CV|PCC|GDE|TCP|MASSA|OKAIDA).*', '', n)
    return n

def process_faction_data_v5():
    kml_path = r'C:\Users\Boanerges\Downloads\ORCRIMS_2026.kml'
    dict_path = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\bairros_centros_latlong.json'
    
    print("1. Carregando Dicionário Oficial de Localidades...")
    with open(dict_path, 'r', encoding='utf-8') as f:
        official_dict = json.load(f)
    
    # Criar lista de nomes oficiais normalizados para match rapido
    official_names = {normalize_text(name): name for name in official_dict.keys()}
    centers = []
    for name, coords in official_dict.items():
        centers.append({'name': name, 'lat': coords['lat'], 'long': coords['long']})

    print("2. Extraindo dados do KML...")
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    micronodos_list = []
    folders = root.findall('.//kml:Folder', ns)
    for folder in folders:
        folder_name_elem = folder.find('kml:name', ns)
        if folder_name_elem is None: continue
        fname = folder_name_elem.text.upper()
        
        faction = "NEUTRO"
        if "COMANDO VERMELHO" in fname or " CV " in fname: faction = "CV"
        elif "TCP" in fname or "GDE" in fname or "GUARDIÕES" in fname: faction = "TCP/GDE"
        elif "PRIMEIRO COMANDO" in fname or "PCC" in fname: faction = "PCC"
        elif "MASSA" in fname: faction = "MASSA"
        elif "OKAIDA" in fname: faction = "OKAIDA"
        elif "DISPUTA" in fname: faction = "DISPUTA"
        
        if faction == "NEUTRO": continue
        
        for pm in folder.findall('.//kml:Placemark', ns):
            name_raw = pm.find('kml:name', ns).text if pm.find('kml:name', ns) is not None else "S/N"
            
            # Extrair coordenadas para geolocalizacao
            lat, lon = None, None
            coords_elem = pm.find('.//kml:coordinates', ns)
            if coords_elem is not None and coords_elem.text:
                try:
                    c = coords_elem.text.strip().split()[0].split(',')
                    lon, lat = float(c[0]), float(c[1])
                except: pass
            
            # Identificar Local Oficial
            area_identificada = "DESCONHECIDO"
            norm_name = normalize_text(name_raw)
            
            # Tenta Match por Nome
            if norm_name in official_names:
                area_identificada = official_names[norm_name]
            else:
                # Tenta Match por geolocalizacao (ponto mais proximo)
                if lat is not None and lon is not None:
                    min_dist = float('inf')
                    for c in centers:
                        d = haversine(lon, lat, c['long'], c['lat'])
                        if d < min_dist:
                            min_dist = d
                            area_identificada = c['name']
            
            micronodos_list.append({
                'micronodo': name_raw,
                'area_oficial': area_identificada,
                'faction': faction,
                'lat': lat,
                'long': lon
            })

    print(f"✅ {len(micronodos_list)} micronodos processados.")
    df_micro = pd.DataFrame(micronodos_list)

    print("3. Agregando por Bairros/Cidades Oficiais...")
    # Agrupar pela area_oficial e pegar a faccao majoritaria
    counts = df_micro.groupby(['area_oficial', 'faction']).size().reset_index(name='n')
    # Ordenar por 'n' e remover duplicatas de area_oficial (fica o maior n)
    df_agregado = counts.sort_values('n', ascending=False).drop_duplicates('area_oficial')

    # Garantir que todas as 305 áreas existam na tabela final (preencher com NEUTRO se nao houver dado)
    final_rows = []
    for official_name in official_dict.keys():
        row_data = df_agregado[df_agregado['area_oficial'] == official_name]
        if not row_data.empty:
            final_rows.append({
                'local': official_name,
                'faccao_predominante': row_data.iloc[0]['faction'],
                'grau_dominio': 0.85
            })
        else:
            final_rows.append({
                'local': official_name,
                'faccao_predominante': 'NEUTRO',
                'grau_dominio': 0.0
            })
    
    df_final = pd.DataFrame(final_rows)

    print("4. Salvando...")
    output_dir = r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\inteligencia'
    os.makedirs(output_dir, exist_ok=True)
    
    # Arquivo 1: Micronodos Completo
    df_micro.to_csv(os.path.join(output_dir, 'micronodos_faccoes_2026.csv'), index=False, encoding='utf-8')
    
    # Arquivo 2: Bairros Agregado (Exatamente as 305 chaves)
    df_final.to_csv(os.path.join(output_dir, 'bairros_faccoes_2026.csv'), index=False, encoding='utf-8')

    # Sincronização Principal
    df_final.to_csv(r'C:\Users\Boanerges\Desktop\Projetos\Report Preview\data\raw\inteligencia_faccoes.csv', index=False, encoding='utf-8')

    print(f"🚀 SUCESSO!")
    print(f"- Micronodos: {len(df_micro)} registros.")
    print(f"- Bairros/Cidades: {len(df_final)} registros oficiais mapeados.")
    print(f"- Sistema atualizado com facções 2026.")

if __name__ == "__main__":
    process_faction_data_v5()

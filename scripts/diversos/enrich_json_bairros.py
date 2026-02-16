import json
import os
import numpy as np
from scipy.spatial import KDTree
import unicodedata

# Caminhos
BASE_DIR = os.getcwd()
JSON_SOURCE = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais.json')
JSON_OUTPUT = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.json')
BAIRROS_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')

def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def enrich():
    print("--- INICIANDO ENRIQUECIMENTO ESPACIAL DO JSON (V4 - HYBRID STRUCTURE) ---")
    
    with open(BAIRROS_FILE, 'r', encoding='utf-8') as f:
        geo_ref = json.load(f)
    
    node_names = []
    node_coords = []
    for name, info in geo_ref.items():
        node_names.append(name)
        node_coords.append([float(info['lat']), float(info['long'])])
    
    tree = KDTree(node_coords)
    print(f"Referencia: {len(node_names)} localidades.")

    with open(JSON_SOURCE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Itens no JSON raiz: {len(raw_data)}")
    
    count_enriched = 0
    count_already_had = 0
    count_no_coords = 0
    total_found = 0

    def process_item(item):
        nonlocal count_enriched, count_already_had, count_no_coords, total_found
        if not isinstance(item, dict): return
        
        # Filtro de metadados
        if 'id' not in item and 'tipo_evento' not in item: return
        
        total_found += 1
        b_at = item.get('bairro')
        if b_at and str(b_at).lower() not in ["null", "none", ""]:
            item['bairro'] = normalize_text(b_at)
            count_already_had += 1
            return
            
        try:
            lat = float(item.get('latitude', 0))
            lon = float(item.get('longitude', 0))
            if lat != 0 and lon != 0:
                dist, idx = tree.query([lat, lon])
                if dist < 0.05: 
                    item['bairro'] = node_names[idx]
                    count_enriched += 1
                else: count_no_coords += 1
            else: count_no_coords += 1
        except: count_no_coords += 1

    for entry in raw_data:
        if not isinstance(entry, dict): continue
        
        # Caso 1: Item e uma ocorrencia direta
        if 'id' in entry or 'tipo_evento' in entry:
            process_item(entry)
        
        # Caso 2: Item e um bloco que contem lista de ocorrencias
        elif 'data' in entry and isinstance(entry['data'], list):
            for sub_item in entry['data']:
                process_item(sub_item)

    print("Concluido:")
    print(f"   - Total ocorrencias detectadas: {total_found}")
    print(f"   - Ja possuiam bairro: {count_already_had}")
    print(f"   - Bairros atribuidos via GPS: {count_enriched}")
    print(f"   - Sem coordenadas/fora do range: {count_no_coords}")

    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)
    
    print(f"Arquivo salvo em: {JSON_OUTPUT}")

if __name__ == "__main__":
    enrich()

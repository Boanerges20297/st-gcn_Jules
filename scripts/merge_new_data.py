import json
import os
import re
import numpy as np
from scipy.spatial import KDTree
import unicodedata

# Configurações de Caminhos
BASE_DIR = os.getcwd()
OFFICIAL_BASE = os.path.join(BASE_DIR, 'data', 'raw', 'dados_status_ocorrencias_gerais_ENRIQUECIDO.json')
BAIRROS_REF = os.path.join(BASE_DIR, 'data', 'raw', 'bairros_centros_latlong.json')

def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def robust_load_new_data(path):
    """Extrai objetos JSON de forma bruta para lidar com arquivos malformados do PHPMyAdmin."""
    print(f"Lendo arquivo bruto: {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Tenta carregar como JSON padrão primeiro
    try:
        data = json.loads(content)
        if isinstance(data, list): return data
    except:
        print("Aviso: Formato JSON padrão falhou. Iniciando extração por blocos de texto (Brute Force)...")

    # 2. Extração via Regex (Busca qualquer padrão { ... "id": "..." ... })
    # Captura objetos que tenham ID e Data (para garantir que sejam registros de crime)
    pattern = r'\{[^{}]*?"id":\s*?"\d+"[^{}]*?\}'
    matches = re.findall(pattern, content, re.DOTALL)
    
    records = []
    for m in matches:
        try:
            # Limpa possíveis vírgulas extras ou caracteres de controle
            clean_m = m.strip().rstrip(',')
            obj = json.loads(clean_m)
            
            # Se o objeto tiver um campo 'data' que é outro dicionário (caso aninhado)
            if 'data' in obj and isinstance(obj['data'], dict):
                obj = obj['data']
                
            records.append(obj)
        except:
            continue
            
    return records

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def merge(new_data_path):
    print(f"--- INICIANDO MESCLAGEM ROBUSTA: {new_data_path} ---")
    
    # 1. Carregar Base Oficial e criar set de chaves compostas para unicidade
    if os.path.exists(OFFICIAL_BASE):
        with open(OFFICIAL_BASE, 'r', encoding='utf-8') as f:
            official_data = json.load(f)
        print(f"Base oficial carregada: {len(official_data)} registros.")
    else:
        official_data = []
        print("Aviso: Base oficial nao encontrada. Criando uma nova.")

    # Chave de unicidade: ID + DATA + HORA (para evitar conflitos de IDs reaproveitados)
    existing_keys = set()
    for item in official_data:
        if isinstance(item, dict):
            key = f"{item.get('id')}_{item.get('data')}_{item.get('hora')}"
            existing_keys.add(key)

    # 2. Carregar Referencia Geo para enriquecimento espacial
    geo_ref = {}
    if os.path.exists(BAIRROS_REF):
        with open(BAIRROS_REF, 'r', encoding='utf-8') as f:
            geo_ref = json.load(f)
            
    node_names = []
    node_coords = []
    for name, info in geo_ref.items():
        node_names.append(name)
        node_coords.append([float(info['lat']), float(info['long'])])
    
    tree = KDTree(node_coords) if node_coords else None

    # 3. Extração e Processamento
    new_records = robust_load_new_data(new_data_path)
    to_add = []
    count_new = 0
    count_dupes = 0
    count_enriched = 0

    for item in new_records:
        if not isinstance(item, dict) or 'id' not in item: continue
        
        # Gera chave de unicidade
        item_key = f"{item.get('id')}_{item.get('data')}_{item.get('hora')}"
        
        if item_key in existing_keys:
            count_dupes += 1
            continue

        # Enriquecimento Espacial (GPS -> Bairro)
        b_at = item.get('bairro')
        if tree and (not b_at or str(b_at).lower() in ["null", "none", ""]):
            try:
                lat, lon = float(item.get('latitude', 0)), float(item.get('longitude', 0))
                if lat != 0:
                    dist, idx = tree.query([lat, lon])
                    # Raio de aprox 5km
                    if dist < 0.05:
                        item['bairro_geo'] = node_names[idx]
                        count_enriched += 1
            except: pass
        else:
            item['bairro'] = normalize_text(b_at)

        to_add.append(item)
        existing_keys.add(item_key)
        count_new += 1

    # 4. Mesclar e Salvar
    if to_add:
        final_data = official_data + to_add
        save_json(final_data, OFFICIAL_BASE)
        print(f"Sucesso!")
        print(f"   - Novos registros inseridos: {count_new}")
        print(f"   - Registros enriquecidos via GPS: {count_enriched}")
        print(f"   - Duplicatas ignoradas: {count_dupes}")
        print(f"Base total atualizada para {len(final_data)} registros em {OFFICIAL_BASE}")
    else:
        print("Nenhum registro novo para inserir.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python scripts/merge_new_data.py caminho_do_novo_arquivo.json")
    else:
        merge(sys.argv[1])

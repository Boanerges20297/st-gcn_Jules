import json
import re
import os

def robust_json_extractor(file_path):
    """Lê um arquivo JSON gigante e malformado, extraindo objetos individuais."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procura por padrões de objetos { ... } que tenham a chave "id" ou "id_evento"
    # Usamos uma abordagem de encontrar blocos entre chaves
    print(f"Lendo {file_path} ({len(content)} bytes)...")
    
    # Tenta carregar como lista primeiro (caso padrao)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except:
        print("Aviso: JSON malformado detectado. Iniciando extração por blocos...")

    # Se falhar, extraímos todos os dicionários válidos manualmente
    # Procura por strings que começam com { e terminam com } e contêm "id":
    records = []
    # Regex para capturar objetos JSON (aproximado, mas funcional para este caso)
    # Procuramos por objetos que contenham "id": "..."
    matches = re.finditer(r'\{[^{}]*?"id":\s*?"(\d+)"[^{}]*?\}', content, re.DOTALL)
    
    for match in matches:
        try:
            obj = json.loads(match.group(0))
            records.append(obj)
        except:
            continue
            
    # Caso haja listas aninhadas em listas (como vimos no debug)
    final_records = []
    def flatten(item):
        if isinstance(item, list):
            for i in item: flatten(i)
        elif isinstance(item, dict):
            # Se o dicionário tem um campo que parece outro registro, mergulha
            if 'data' in item and isinstance(item['data'], dict):
                final_records.append(item['data'])
            else:
                final_records.append(item)
                
    flatten(records)
    return final_records

def perform_mega_merge():
    raw_path = 'data/raw/dados_status_ocorrencias_gerais.json'
    enriched_path = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
    
    # 1. Extração robusta
    all_raw = robust_json_extractor(raw_path)
    print(f"Registros extraídos do RAW: {len(all_raw)}")
    
    # 2. Carrega Enriquecido Atual
    with open(enriched_path, 'r', encoding='utf-8') as f:
        enriched_data = json.load(f)
    
    existing_keys = set()
    for item in enriched_data:
        if isinstance(item, dict):
            key = f"{item.get('id')}_{item.get('data')}_{item.get('hora')}"
            existing_keys.add(key)
            
    # 3. Merge
    new_count = 0
    for rec in all_raw:
        key = f"{rec.get('id')}_{rec.get('data')}_{rec.get('hora')}"
        if key not in existing_keys:
            enriched_data.append(rec)
            existing_keys.add(key)
            new_count += 1
            
    print(f"Novos registros únicos adicionados: {new_count}")
    
    # 4. Salva Versão Final
    with open(enriched_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Mega Merge Concluído! Total na base: {len(enriched_data)}")

if __name__ == "__main__":
    perform_mega_merge()

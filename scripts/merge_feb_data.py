import json
import os
from datetime import datetime

def merge_february_data():
    raw_path = 'data/raw/dados_status_ocorrencias_gerais.json'
    enriched_path = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
    
    if not os.path.exists(raw_path):
        print("Arquivo raw nao encontrado.")
        return

    print("Carregando arquivo RAW...")
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print("Carregando arquivo ENRIQUECIDO...")
    with open(enriched_path, 'r', encoding='utf-8') as f:
        enriched_data = json.load(f)
    
    # Conjunto de IDs ja existentes para evitar duplicidade
    existing_ids = set()
    for item in enriched_data:
        if isinstance(item, dict) and item.get('id'):
            existing_ids.add(str(item.get('id')))
    
    new_records = []
    skipped = 0
    
    print("Analisando registros para merge...")
    for item in raw_data:
        target = item
        # Caso o registro esteja aninhado (como vimos no debug)
        if isinstance(item, dict) and 'data' in item and isinstance(item['data'], dict):
            target = item['data']
        
        if not isinstance(target, dict):
            continue
            
        # Verifica se e de Fevereiro/2026
        dt_str = str(target.get('data', ''))
        if '2026-02' in dt_str:
            rec_id = str(target.get('id', ''))
            if rec_id not in existing_ids:
                new_records.append(target)
                existing_ids.add(rec_id)
            else:
                skipped += 1
                
    print(f"Novos registros encontrados: {len(new_records)}")
    print(f"Registros ignorados (duplicados): {skipped}")
    
    if len(new_records) > 0:
        # Merge
        final_data = enriched_data + new_records
        with open(enriched_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print(f"✅ Merge concluido! Total na base enriquecida: {len(final_data)}")
    else:
        print("Nenhum registro novo de fevereiro para mergear.")

if __name__ == "__main__":
    merge_february_data()

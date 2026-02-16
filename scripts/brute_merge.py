import json
import re
import os

def brute_force_extract(file_path):
    print(f"Brute force extract from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procura por qualquer bloco { ... "data": "2026-02 ... }
    # Vamos capturar o objeto inteiro entre as chaves {}
    # Usamos uma regex gananciosa para o conteudo mas limitada pelo fechamento
    # Isso assume que nao ha objetos aninhados complexos (o que parece ser o caso nos registros de crime)
    pattern = r'\{[^{}]*?"id":\s*?"\d+"[^{}]*?"data":\s*?"2026-02-[^{}]*?\}'
    matches = re.findall(pattern, content, re.DOTALL)
    
    records = []
    for m in matches:
        try:
            # Limpa possíveis vírgulas extras no final do bloco se houver
            clean_m = m.strip().rstrip(',')
            obj = json.loads(clean_m)
            records.append(obj)
        except Exception as e:
            continue
    
    print(f"Registros de Fevereiro encontrados via Brute Force: {len(records)}")
    return records

def perform_brute_merge():
    raw_path = 'data/raw/dados_status_ocorrencias_gerais.json'
    enriched_path = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
    
    feb_records = brute_force_extract(raw_path)
    
    with open(enriched_path, 'r', encoding='utf-8') as f:
        enriched_data = json.load(f)
    
    existing_keys = {f"{item.get('id')}_{item.get('data')}_{item.get('hora')}" for item in enriched_data if isinstance(item, dict)}
    
    added = 0
    for rec in feb_records:
        key = f"{rec.get('id')}_{rec.get('data')}_{rec.get('hora')}"
        if key not in existing_keys:
            enriched_data.append(rec)
            existing_keys.add(key)
            added += 1
            
    if added > 0:
        with open(enriched_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        print(f"✅ SUCESSO! {added} novos registros de Fevereiro adicionados via Brute Force.")
    else:
        print("Nenhum registro novo de Fevereiro encontrado (ja estao na base ou erro de busca).")

if __name__ == "__main__":
    perform_brute_merge()

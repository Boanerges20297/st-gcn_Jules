import json
p = 'data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.json'
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
updated = 0
for item in data:
    if isinstance(item, dict) and 'bairro_geo' in item:
        bg = item.get('bairro_geo')
        b = item.get('bairro')
        if bg and (not b or str(b).strip().lower() in ('', 'null', 'none')):
            item['bairro'] = bg
            updated += 1
with open(p, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print('UPDATED', updated)

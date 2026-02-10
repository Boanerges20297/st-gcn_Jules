import json

data = json.load(open('data/static/nodes_polygons.geojson', encoding='utf-8'))
feat = data['features'][0]
print(f"First feature: {feat['properties'].get('name', feat['properties'].get('node_id'))}")
print(f"Props keys: {list(feat['properties'].keys())[:10]}")

# Contar bairros vs cidades
count_bairro = 0
count_cidade = 0
for feat in data['features']:
    node_type = feat['properties'].get('node_type')
    if node_type == 'bairro':
        count_bairro += 1
    elif node_type == 'cidade':
        count_cidade += 1
        
print(f'\nTotal features: {len(data["features"])}')
print(f'Bairros: {count_bairro}')
print(f'Cidades: {count_cidade}')

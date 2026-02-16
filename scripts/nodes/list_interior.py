import json

with open('data/raw/bairros_centros_latlong.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

interior = [name for name, info in data.items() if info.get('regiao') == 'interior']
interior.sort()
print(f"Total Interior: {len(interior)}")
for i, name in enumerate(interior):
    print(f"{i+1}: {name}")

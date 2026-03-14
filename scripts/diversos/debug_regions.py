#!/usr/bin/env python3
import json

# Verificar cada arquivo regional
regions = ['capital', 'rmf', 'interior']
for region in regions:
    file_path = f'outputs/top20_micro_nodes_{region}.geojson'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        count = len(data['features'])
        print(f"\n{region.upper()}: {count} features")
        # Mostrar primeiras 5
        for feat in data['features'][:5]:
            name = feat['properties']['name']
            mun = feat['properties']['municipality']
            rank = feat['properties']['rank']
            print(f"  {rank}. {name} ({mun})")
    except Exception as e:
        print(f"\n{region.upper()}: ERRO - {e}")

# Verificar também o arquivo consolidado
print("\n" + "="*70)
print("ARQUIVO CONSOLIDADO (top20_micro_nodes.geojson)")
print("="*70)
with open('outputs/top20_micro_nodes.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

dist = {'capital': 0, 'rmf': 0, 'interior': 0}
for feat in data['features']:
    region = feat['properties']['region']
    if region in dist:
        dist[region] += 1

print(f"Distribuição: {dist}")
print(f"Total: {sum(dist.values())}")

# Mostrar amostras de RMF
print("\nAmostras de RMF no consolidado:")
rmf_features = [f for f in data['features'] if f['properties']['region'] == 'rmf']
if rmf_features:
    for feat in rmf_features[:10]:
        name = feat['properties']['name']
        mun = feat['properties']['municipality']
        print(f"  - {name} ({mun})")
else:
    print("  NENHUM!")

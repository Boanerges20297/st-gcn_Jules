#!/usr/bin/env python3
import json

# Verificar detalhes dos features
file_path = 'outputs/top20_micro_nodes_rmf.geojson'

print("Analisando RMF GeoJSON...")
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data.get('features', [])
print(f"Total features: {len(features)}\n")

# Pegas primeiras 3
for i, feat in enumerate(features[:3]):
    print(f"Feature {i+1}:")
    print(f"  name: {feat['properties'].get('name')}")
    print(f"  region: {feat['properties'].get('region')}")
    print(f"  geometry type: {feat.get('geometry', {}).get('type')}")
    coords = feat.get('geometry', {}).get('coordinates', [])
    print(f"  coordinates: {coords}")
    print()

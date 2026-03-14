#!/usr/bin/env python3
import json

# Verificar distribuição de municípios nos arquivos
for region in ['capital', 'rmf', 'interior']:
    file_path = f'outputs/top20_micro_nodes_{region}.geojson'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    munics = {}
    for feat in data['features']:
        mun = feat['properties'].get('municipality', 'NONE')
        munics[mun] = munics.get(mun, 0) + 1
    
    print(f"\n{region.upper()} - Municípios:")
    for k, v in sorted(munics.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}")

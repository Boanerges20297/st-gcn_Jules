#!/usr/bin/env python3
import json
import requests

# Testar a API
BASE_URL = 'http://localhost:5000'

regions = ['fortaleza', 'rmf', 'interior', 'all']

for region in regions:
    url = f'{BASE_URL}/api/top20_micro_nodes?region={region}'
    print(f"\n{'='*70}")
    print(f"Testando: {region}")
    print(f"URL: {url}")
    print(f"{'='*70}")
    
    try:
        resp = requests.get(url, timeout=5)
        if resp.ok:
            data = resp.json()
            features = data.get('features', [])
            print(f"✓ Recebido: {len(features)} features")
            
            # Contar por região (se aplicável)
            if region == 'all':
                dist = {'capital': 0, 'rmf': 0, 'interior': 0}
                for feat in features:
                    r = feat['properties'].get('region', 'unknown')
                    if r in dist:
                        dist[r] += 1
                print(f"  Distribuição: {dist}")
            
            # Mostrar primeiras 3
            for feat in features[:3]:
                name = feat['properties']['name']
                mun = feat['properties']['municipality']
                print(f"  - {name} ({mun})")
        else:
            print(f"✗ Erro: {resp.status_code}")
    except Exception as e:
        print(f"✗ Exceção: {e}")

print(f"\n{'='*70}")

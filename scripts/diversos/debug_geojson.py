#!/usr/bin/env python3
import json

# Verificar a estrutura dos arquivos GeoJSON
regions = ['capital', 'rmf', 'interior']

print("="*70)
print("VERIFICANDO ESTRUTURA DOS GEOJSON")
print("="*70)

for region in regions:
    file_path = f'outputs/top20_micro_nodes_{region}.geojson'
    print(f"\n{region.upper()}: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get('features', [])
        print(f"  Total features: {len(features)}")
        
        if features:
            feat = features[0]
            print(f"  Propriedades: {list(feat['properties'].keys())}")
            print(f"  Amostra - name: {feat['properties'].get('name')}")
            print(f"  Amostra - region: {feat['properties'].get('region')}")
            print(f"  Amostra - region_type: {feat['properties'].get('region_type')}")
            
            # Verificar distribuição de propriedade 'region'
            regions_in_data = {}
            for feat in features:
                r = feat['properties'].get('region', 'MISSING')
                regions_in_data[r] = regions_in_data.get(r, 0) + 1
            
            print(f"  Valores de 'region' encontrados: {regions_in_data}")
    
    except Exception as e:
        print(f"  ERRO: {e}")

# Também verificar se há polygon files
print("\n" + "="*70)
print("VERIFICANDO ARQUIVOS DE POLÍGONOS")
print("="*70)

polygon_file = 'data/geojson_bairros_ceara.geojson'
try:
    with open(polygon_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get('features', [])
    print(f"\nPolígonos: {len(features)} features")
    
    if features:
        feat = features[0]
        print(f"  Propriedades: {list(feat['properties'].keys())}")
        
        # Verificar region_type
        region_types = {}
        for feat in features[:100]:  # Amostra dos primeiros 100
            rt = feat['properties'].get('region_type', 'MISSING')
            region_types[rt] = region_types.get(rt, 0) + 1
        
        print(f"  Valores de 'region_type' encontrados: {region_types}")
        
except Exception as e:
    print(f"  Erro ao ler polígonos: {e}")

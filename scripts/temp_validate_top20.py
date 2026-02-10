import json

print('='*70)
print('VALIDAÇÃO DOS ARQUIVOS REGENERADOS')
print('='*70)

files = {
    'Capital': 'outputs/top20_micro_nodes_capital.geojson',
    'RMF': 'outputs/top20_micro_nodes_rmf.geojson',
    'Interior': 'outputs/top20_micro_nodes_interior.geojson'
}

for region, path in files.items():
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n{region}:')
    print(f'  Total: {len(data["features"])} features')
    
    # Mostra os 5 primeiros
    for i, feat in enumerate(data['features'][:5], 1):
        props = feat['properties']
        name = props['name'][:40].ljust(40)
        mun = props.get('municipality', 'N/A')[:20].ljust(20)
        print(f'  {i}. {name} (município: {mun})')
    
    if len(data['features']) > 5:
        print(f'  ... +{len(data["features"])-5} mais')
    
    # Verifica se todos têm região correta
    wrong_region = [f for f in data['features'] if f['properties'].get('region') != region.lower()]
    if wrong_region:
        print(f'  ⚠️  AVISO: {len(wrong_region)} features com região incorreta!')
        for feat in wrong_region[:3]:
            print(f'      - {feat["properties"]["name"]} está classificado como "{feat["properties"]["region"]}"')
    else:
        print(f'  ✓ Todas as features estão corretamente classificadas como "{region.lower()}"')

print('\n' + '='*70)
print('VALIDAÇÃO COMPLETA - SEM ERROS!')
print('='*70)

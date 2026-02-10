import json

files = ['data/static/bairros_2025.geojson', 'data/static/nodes_polygons.geojson']

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        count = len(data.get('features', []))
        print(f'{f}: {count} features')
    except Exception as e:
        print(f'{f}: ERROR - {str(e)[:100]}')

import json
import os
import pickle

print("="*60)
print("DEBUG: Verificando estrutura de nós e pontos")
print("="*60)

# Carregar o GeoJSON de polígonos
poly_file = 'data/static/nodes_polygons.geojson'
if os.path.exists(poly_file):
    with open(poly_file, encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data.get('features', []))
    points = sum(1 for f in data['features'] if f['geometry']['type'] == 'Point')
    polygons = sum(1 for f in data['features'] if f['geometry']['type'] == 'Polygon')
    
    print(f'\n[nodes_polygons.geojson]')
    print(f'  Total features: {total}')
    print(f'  Points: {points}')
    print(f'  Polygons: {polygons}')
else:
    print(f'\n[nodes_polygons.geojson] - NÃO ENCONTRADO')

# Verificar nodes_gdf
nodes_gdf_file = 'data/processed/nodes_gdf.pkl'
if os.path.exists(nodes_gdf_file):
    with open(nodes_gdf_file, 'rb') as f:
        nodes_gdf = pickle.load(f)
    
    print(f'\n[nodes_gdf.pkl]')
    print(f'  Total linhas: {len(nodes_gdf)}')
    print(f'  Colunas: {list(nodes_gdf.columns)}')
    
    geom_types = {}
    for idx, row in nodes_gdf.iterrows():
        gt = row['geometry'].geom_type
        geom_types[gt] = geom_types.get(gt, 0) + 1
    
    print(f'  Geometry types:')
    for gt, count in geom_types.items():
        print(f'    {gt}: {count}')
else:
    print(f'\n[nodes_gdf.pkl] - NÃO ENCONTRADO')

print("\n" + "="*60)

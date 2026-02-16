"""
Validação do TCP - Extrai coordenadas e bairros contidos
Similar à validação feita para o COMANDO VERMELHO

Script de análise detalhada para TCP (TERCEIRO COMANDO PURO)
"""

import json
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

def validate_tcp_territory():
    """Extrai e valida o território do TCP"""
    
    print("\n" + "="*70)
    print("🔍 VALIDAÇÃO: TCP - TERCEIRO COMANDO PURO")
    print("="*70)
    
    # Carregar GeoJSON do TCP
    tcp_file = BASE_DIR / 'data' / 'raw' / 'inteligencia' / 'TERCEIRO COMANDO PURO.geojson'
    
    if not tcp_file.exists():
        print(f"❌ Arquivo não encontrado: {tcp_file}")
        return
    
    print(f"\n📂 Carregando: TERCEIRO COMANDO PURO.geojson")
    tcp_gdf = gpd.read_file(tcp_file)
    print(f"✅ Carregado: {len(tcp_gdf)} features")
    
    # Extrair info dos polígonos
    print(f"\n📊 ANÁLISE DOS POLÍGONOS:")
    total_polys = 0
    geom_types = {}
    
    for idx, row in tcp_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        
        geom_type = geom.geom_type
        geom_types[geom_type] = geom_types.get(geom_type, 0) + 1
        
        if geom_type == 'Polygon':
            total_polys += 1
        elif geom_type == 'MultiPolygon':
            total_polys += len(list(geom.geoms))
    
    print(f"  Tipos de geometria encontrados:")
    for gtype, count in geom_types.items():
        print(f"    - {gtype}: {count}")
    print(f"  Total de polígonos: {total_polys}")
    
    # Extrair coordenadas e calcular bounds
    print(f"\n🗺️  COORDENADAS DOS POLÍGONOS:")
    all_coords = []
    
    for idx, row in tcp_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            all_coords.extend(coords)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                all_coords.extend(coords)
    
    if all_coords:
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        
        print(f"  Bounds de Longitude: {min(lons):.6f} a {max(lons):.6f}")
        print(f"  Bounds de Latitude:  {min(lats):.6f} a {max(lats):.6f}")
        print(f"  Total de coordenadas: {len(all_coords)}")
    
    # Carregar bairros e cidades
    bairros_file = BASE_DIR / 'data' / 'raw' / 'bairros_centros_latlong.json'
    
    print(f"\n📋 Carregando bairros e cidades...")
    with open(bairros_file, 'r', encoding='utf-8') as f:
        bairros_data = json.load(f)
    
    locations = []
    for name, info in bairros_data.items():
        if name in ["Nome", "null", "None", ""] or name is None:
            continue
        location = {
            'name': name,
            'lat': info['lat'],
            'lng': info['long'],
            'regiao': info.get('regiao', 'desconhecido').lower(),
            'node_type': 'bairro' if info.get('regiao', '').lower() in ['fortaleza', 'rmf'] else 'cidade'
        }
        locations.append(location)
    
    print(f"✅ Carregados: {len(locations)} locais")
    
    # Realizar análise espacial
    print(f"\n🔎 ANÁLISE ESPACIAL:")
    
    # Unir todos os polígonos
    polygons = []
    for idx, row in tcp_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == 'Polygon':
            polygons.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                polygons.append(poly)
    
    combined_poly = unary_union(polygons)
    print(f"  Polígonos unificados: ✅")
    
    # Análise de pontos
    inside_polygon = []
    near_boundary = []
    outside = []
    
    for location in locations:
        point = Point(location['lng'], location['lat'])
        
        if combined_poly.contains(point):
            inside_polygon.append(location)
        else:
            distance = point.distance(combined_poly.boundary)
            if distance < 0.01:  # ~1km
                near_boundary.append(location)
            else:
                outside.append(location)
    
    print(f"\n  Resultado:")
    print(f"    Dentro do polígono (✓):  {len(inside_polygon)} locais")
    print(f"    Perto da borda (~):      {len(near_boundary)} locais")
    print(f"    Fora do polígono:        {len(outside)} locais")
    
    # Breakdown por tipo e região
    print(f"\n  Dentro do Polígono (Alta Confiança):")
    print(f"    Bairros Fortaleza: {len([l for l in inside_polygon if l['node_type']=='bairro' and l['regiao']=='fortaleza'])}")
    print(f"    Bairros RMF: {len([l for l in inside_polygon if l['node_type']=='bairro' and l['regiao']=='rmf'])}")
    print(f"    Cidades: {len([l for l in inside_polygon if l['node_type']=='cidade'])}")
    
    print(f"\n  Perto da Borda (Confiança Média):")
    print(f"    Bairros Fortaleza: {len([l for l in near_boundary if l['node_type']=='bairro' and l['regiao']=='fortaleza'])}")
    print(f"    Bairros RMF: {len([l for l in near_boundary if l['node_type']=='bairro' and l['regiao']=='rmf'])}")
    print(f"    Cidades: {len([l for l in near_boundary if l['node_type']=='cidade'])}")
    
    # Lista detalhada
    total_assigned = inside_polygon + near_boundary
    
    print(f"\n{'='*70}")
    print(f"📌 BAIRROS/CIDADES ATRIBUÍDOS AO TCP ({len(total_assigned)} total)")
    print(f"{'='*70}")
    
    print(f"\n✓ DENTRO DO POLÍGONO ({len(inside_polygon)}):")
    fortaleza_inside = sorted([l['name'] for l in inside_polygon if l['regiao']=='fortaleza'])
    if fortaleza_inside:
        for name in fortaleza_inside:
            print(f"   {name}")
    else:
        print("   (nenhum)")
    
    if len([l for l in inside_polygon if l['regiao']=='rmf']) > 0:
        print(f"\n   RMF:")
        for name in sorted([l['name'] for l in inside_polygon if l['regiao']=='rmf']):
            print(f"   {name}")
    
    print(f"\n~ PERTO DA BORDA ({len(near_boundary)}):")
    fortaleza_boundary = sorted([l['name'] for l in near_boundary if l['regiao']=='fortaleza'])
    if fortaleza_boundary:
        for name in fortaleza_boundary:
            print(f"   {name}")
    else:
        print("   (nenhum)")
    
    if len([l for l in near_boundary if l['regiao']=='rmf']) > 0:
        print(f"\n   RMF:")
        for name in sorted([l['name'] for l in near_boundary if l['regiao']=='rmf']):
            print(f"   {name}")
    
    # Salvar resultado
    print(f"\n{'='*70}")
    print(f"💾 SALVANDO RESULTADO")
    print(f"{'='*70}")
    
    result = {
        'faction': 'TCP',
        'total_assigned': len(total_assigned),
        'inside_polygon': len(inside_polygon),
        'near_boundary': len(near_boundary),
        'bairros_fortaleza': len([l for l in total_assigned if l['node_type']=='bairro' and l['regiao']=='fortaleza']),
        'bairros_rmf': len([l for l in total_assigned if l['node_type']=='bairro' and l['regiao']=='rmf']),
        'cidades': len([l for l in total_assigned if l['node_type']=='cidade']),
        'assignments': [
            {
                'name': l['name'],
                'in_polygon': l in inside_polygon,
                'regiao': l['regiao'],
                'node_type': l['node_type']
            }
            for l in total_assigned
        ]
    }
    
    output_file = BASE_DIR / 'outputs' / 'tcp_validation.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Salvo: {output_file}")
    print(f"\n✅ VALIDAÇÃO CONCLUÍDA!")

if __name__ == '__main__':
    validate_tcp_territory()

#!/usr/bin/env python
"""
Debug: Identificar micro-nós que não aparecem no dashboard/mapa
"""

import pickle
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')

print("=" * 80)
print("🔍 ANALISANDO MICRO-NÓS INVISÍVEIS")
print("=" * 80)

# Carregar dados
if not os.path.exists(data_path):
    print(f"❌ Arquivo não encontrado: {data_path}")
    exit(1)

with open(data_path, 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data.get('nodes_gdf')
if nodes_gdf is None:
    print("❌ nodes_gdf não encontrado no pickle")
    exit(1)

print(f"\n📊 ESTATÍSTICAS GERAIS:")
print(f"   Total de nós: {len(nodes_gdf)}")
print(f"   Colunas: {nodes_gdf.columns.tolist()}")

# Analisar node_type
if 'node_type' in nodes_gdf.columns:
    print(f"\n🏷️ Distribuição por node_type:")
    type_counts = nodes_gdf['node_type'].value_counts()
    for ntype, count in type_counts.items():
        print(f"   {ntype}: {count} nós")
else:
    print("❌ Coluna 'node_type' não encontrada")

# Analisar geometry
print(f"\n🗺️ Verificação de geometria:")
print(f"   Nós com geometry: {nodes_gdf.geometry.notna().sum()}")
print(f"   Nós SEM geometry: {nodes_gdf.geometry.isna().sum()}")

# Analisar se há nós com campos vazios/nulos
print(f"\n📍 Campos críticos para visualização:")
critical_fields = ['name', 'CIDADE', 'node_type', 'geometry', 'faction']
for field in critical_fields:
    if field in nodes_gdf.columns:
        non_null = nodes_gdf[field].notna().sum()
        null_count = len(nodes_gdf) - non_null
        pct = (null_count / len(nodes_gdf)) * 100
        status = "✅" if null_count == 0 else "⚠️"
        print(f"   {status} {field}: {non_null} completos, {null_count} ausentes ({pct:.1f}%)")
    else:
        print(f"   ❌ {field}: NÃO ENCONTRADO")

# Analisar por região
if 'regiao' in nodes_gdf.columns:
    print(f"\n📍 Distribuição por região:")
    region_counts = nodes_gdf['regiao'].value_counts()
    for region, count in region_counts.items():
        print(f"   {region}: {count} nós")

# Nós problemáticos
print(f"\n⚠️ NÓS PROBLEMÁTICOS (que podem não aparecer):")

# 1. Sem nome
no_name = nodes_gdf[nodes_gdf['name'].isna() | (nodes_gdf['name'] == '')]
if len(no_name) > 0:
    print(f"\n   SEM NOME ({len(no_name)} nós):")
    for idx, row in no_name.iterrows():
        print(f"      - node_id={idx}, type={row.get('node_type')}, city={row.get('CIDADE')}")

# 2. Sem geometria válida
no_geom = nodes_gdf[nodes_gdf.geometry.isna()]
if len(no_geom) > 0:
    print(f"\n   SEM GEOMETRIA ({len(no_geom)} nós):")
    for idx, row in no_geom.iterrows():
        print(f"      - node_id={idx}, name={row.get('name')}, type={row.get('node_type')}")

# 3. Geometrias inválidas (fora dos limites do Ceará)
print(f"\n   VERIFICANDO LIMITES GEOGRÁFICOS:")
# Ceará bounds aprox: -7.90 a -2.80 lat, -41.50 a -37.20 lng
valid_gdf = nodes_gdf[nodes_gdf.geometry.notna()].copy()
if len(valid_gdf) > 0:
    valid_gdf['lon'] = valid_gdf.geometry.x
    valid_gdf['lat'] = valid_gdf.geometry.y
    
    # Verificar bounds anômalos
    anomalous = valid_gdf[
        (valid_gdf['lon'] < -42) | (valid_gdf['lon'] > -37) |
        (valid_gdf['lat'] < -8.5) | (valid_gdf['lat'] > -2)
    ]
    
    if len(anomalous) > 0:
        print(f"      ⚠️ {len(anomalous)} nós com coordenadas anômolas:")
        for idx, row in anomalous.iterrows():
            print(f"         - {row['name']}: ({row['lat']:.2f}, {row['lon']:.2f})")
    else:
        print(f"      ✅ Todos os nós estão dentro dos limites esperados")

# 4. Micro-nós específicos
print(f"\n   MICRO-NÓS POR TIPO:")
if 'node_type' in nodes_gdf.columns:
    micro_types = nodes_gdf['node_type'].unique()
    for mtype in sorted(micro_types):
        subset = nodes_gdf[nodes_gdf['node_type'] == mtype]
        has_geom = subset.geometry.notna().sum()
        has_name = (subset['name'].notna() & (subset['name'] != '')).sum()
        print(f"\n      Type: '{mtype}' ({len(subset)} nós)")
        print(f"         - Com geometria: {has_geom}/{len(subset)}")
        print(f"         - Com nome: {has_name}/{len(subset)}")
        
        # Listar alguns exemplos
        samples = subset.head(3)
        for idx, row in samples.iterrows():
            geom_ok = "✓" if row.geometry is not None else "✗"
            name_ok = "✓" if (row['name'] and row['name'] != '') else "✗"
            print(f"            [{geom_ok}] {name_ok} id={idx}: {row['name']} ({row.get('CIDADE', 'N/A')})")

# 5. Verificar se há filtros no dashboard que escondem nós
print(f"\n🔍 POSSÍVEIS CAUSAS DE INVISIBILIDADE:")
print(f"   1. Coluna 'name' vazia/nula → Não aparecem no mapa")
print(f"   2. Geometria nula → Não podem ser renderizadas")
print(f"   3. Fora dos limites de Ceará → Filtro geográfico")
print(f"   4. Sem region_type definido → Pode impedir filtro por região")
print(f"   5. node_type não reconhecido → Possível filtro no JS")

# Resumo
print(f"\n" + "=" * 80)
print(f"📋 RESUMO:")
total_nodes = len(nodes_gdf)
visible_candidates = nodes_gdf[
    (nodes_gdf['name'].notna()) & 
    (nodes_gdf['name'] != '') &
    (nodes_gdf.geometry.notna())
]
invisible = total_nodes - len(visible_candidates)
print(f"   Total: {total_nodes} nós")
print(f"   Potencialmente visíveis: {len(visible_candidates)}")
print(f"   Potencialmente INVISÍVEIS: {invisible}")
print(f"=" * 80)

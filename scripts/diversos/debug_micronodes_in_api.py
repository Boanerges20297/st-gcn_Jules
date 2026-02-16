#!/usr/bin/env python
"""
Debug: Verificar quais nós aparecem no mapa (verificar se API retorna todos)
"""

import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')

with open(data_path, 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data.get('nodes_gdf')

print("=" * 80)
print("🔍 VERIFICANDO APARÊNCIA DOS NÓS NO DASHBOARD")
print("=" * 80)

print("\n📊 Nós BAIRRO (Fortaleza + RMF) - deveriam aparecer sempre:")
bairros = nodes_gdf[nodes_gdf['node_type'] == 'bairro']
print(f"   Total: {len(bairros)}")
print(f"   Regiões presentes: {bairros['regiao'].unique().tolist()}")
if len(bairros) > 0:
    print(f"   Exemplos:")
    for idx, row in bairros.head(5).iterrows():
        print(f"      - id={idx}: {row['name']} ({row['regiao']})")

print("\n📊 Nós CIDADE (Interior + RMF cities) - micro-nós:")
cidades = nodes_gdf[nodes_gdf['node_type'] == 'cidade']
print(f"   Total: {len(cidades)}")
print(f"   Regiões presentes: {cidades['regiao'].unique().tolist()}")
if len(cidades) > 0:
    print(f"   Exemplos de cada região:")
    for regiao in cidades['regiao'].unique():
        subset = cidades[cidades['regiao'] == regiao].head(3)
        print(f"      {regiao}: {subset['name'].tolist()}")

print("\n" + "=" * 80)
print("⚠️ POSSÍVEL PROBLEMA IDENTIFICADO:")
print("=" * 80)
print("""
Os 163 nós do tipo 'cidade' são micro-nós espalhados principalmente no INTERIOR.

Possíveis cenários de invisibilidade:
1️⃣  Se o dashboard filtra por 'region_type', pode não ter este campo 
    → Cria coluna vazia ou com valor padrão?
    
2️⃣  Se há filtro de zoom no mapa:
    → Talvez desapareçam quando zoom < 12?
    
3️⃣  Se há filtro "region_filter":
    → Valores esperados: 'capital', 'rmf', 'interior'
    → O code atual usa 'regiao' de nodes_gdf
    
4️⃣  Se há limite de nós renderizados:
    → App.py renderiza todos os 319?
    → Ou apenas top-K nós?
""")

# Verificar o que app.py faria
print("\n" + "=" * 80)
print("🔍 SIMULANDO O QUE app.py RETORNA:")
print("=" * 80)

# Simular o que app.py faz ao preparar response
print("\n1. Verificar se app.py tem 'region_type' tratado:")
if 'region_type' in nodes_gdf.columns:
    print("   ✅ Coluna 'region_type' encontrada")
else:
    print("   ❌ Coluna 'region_type' NÃO encontrada")
    print("   → app.py pode estar criando essa coluna dinamicamente")

print("\n2. Simulando enrich_regions() logic:")
# Isso é feito em app.py - vamos verificar
nodes_test = nodes_gdf.copy()

# Reproduzir logic de app.py
if 'region_type' not in nodes_test.columns:
    # app.py faz isso:
    # if 'regiao' in nodes_gdf.columns:
    #     nodes_gdf['region_type'] = nodes_gdf['regiao'].replace('fortaleza', 'capital')
    nodes_test['region_type'] = nodes_test['regiao'].replace('fortaleza', 'capital')
    print("   Created 'region_type' from 'regiao'")
    print(f"   ✅ Values: {nodes_test['region_type'].unique().tolist()}")

# Visualizar estrutura final
print("\n3. Estrutura final que vai para frontend (JSON):")
print(f"   Campos que vão no response:")
geojson_fields = ['name', 'node_type', 'regiao', 'region_type', 'faction', 'geometry']
for field in geojson_fields:
    if field in nodes_test.columns:
        print(f"      ✓ {field}")
    else:
        print(f"      ✗ {field} (FALTANDO)")

print("\n" + "=" * 80)
print("💡 RECOMENDAÇÃO:")
print("=" * 80)
print("""
A. Verificar se 'region_type' está sendo criado corretamente no app.py
B. Verificar se template/index.html consegue filtrar por 'interior'
C. Verificar quais nós aparecem quando selecionado 'interior' no filtro
D. Pode estar tudo trabalhando, mas os usuários não sabem que podem 
   filtrar por interior para ver os micro-nós das cidades!
""")

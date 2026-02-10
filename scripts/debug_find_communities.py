#!/usr/bin/env python
"""
Debug: Procurar por 'Morro' e outras comunidades que deveriam estar no grafo
"""

import pickle
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')

with open(data_path, 'rb') as f:
    data = pickle.load(f)

nodes_gdf = data.get('nodes_gdf')

print("=" * 80)
print("🔍 PROCURANDO POR COMUNIDADES/FAVELAS NO GRAFO")
print("=" * 80)

# Procurar por padrões
nome_lower = nodes_gdf['name'].str.lower()

patterns = [
    ('morro', 'Morro', nome_lower.str.contains('morro', regex=False, na=False)),
    ('favela', 'Favela', nome_lower.str.contains('favela', regex=False, na=False)),
    ('comunidade', 'Comunidade', nome_lower.str.contains('comunidade', regex=False, na=False)),
    ('beco', 'Beco', nome_lower.str.contains('beco', regex=False, na=False)),
    ('ouro', 'Ouro', nome_lower.str.contains('ouro', regex=False, na=False)),
]

for pattern, label, mask in patterns:
    count = mask.sum()
    print(f"\n🔍 {label} ({pattern}): {count} encontrados")
    if count > 0:
        matches = nodes_gdf[mask]
        for idx, row in matches.iterrows():
            print(f"   - id={idx}: {row['name']} (type={row['node_type']}, regiao={row['regiao']})")

print("\n" + "=" * 80)
print("⚠️ CONCLUSÃO:")
print("=" * 80)

# Verificar quais estão faltando
geojson_comunidades = [
    "Bairro Mangueiral",
    "Favela da Catita",
    "Sem Terra",
    "Pavuna",
    "Beco da Saudade",
    "Beco Do Coroa",
    "Casarão",
    "Ubaúna - Coreaú",
    "Potira",
    "Zona de Confronto",
    "Favela do Verdão"
]

print(f"\nComunidades do GEOJSON que deveriam estar no grafo:")
for com in geojson_comunidades:
    found = nodes_gdf[nodes_gdf['name'] == com]
    if len(found) > 0:
        print(f"   ✅ {com}")
    else:
        print(f"   ❌ {com} (FALTANDO)")

print("\n" + "=" * 80)

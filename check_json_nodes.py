#!/usr/bin/env python
"""
Verificar quais nós estão no JSON de bairros
"""
import json

with open('data/raw/bairros_centros_latlong.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total de nós no JSON: {len(data)}")
print("\nProcurando por Morro/Favela/Comunidade...")
found = []
for name in data.keys():
    if any(x in name.lower() for x in ['morro', 'favela', 'comunidade', 'beco']):
        found.append(name)

if found:
    print(f"✅ Encontrados {len(found)}:")
    for name in found[:10]:
        print(f"   - {name}")
    if len(found) > 10:
        print(f"   ... e {len(found) - 10} mais")
else:
    print("❌ NENHUM encontrado")

print("\n" + "="*80)
print("Primeiras 20 nós no arquivo:")
for i, name in enumerate(list(data.keys())[:20]):
    regiao = data[name].get('regiao')
    print(f"   {i+1}. {name} ({regiao})")

#!/usr/bin/env python
"""
Procurar em todos os dados por "Morro do Ouro"
"""
import json
import os
from pathlib import Path

search_terms = ["Morro do Ouro", "Morro do Oitão", "Moura brasil"]

print("🔍 Procurando em arquivos JSON/GeoJSON...")
print("=" * 80)

found = []

for filepath in Path('data/').rglob('*.*'):
    if filepath.suffix not in ['.json', '.geojson']:
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for term in search_terms:
                if term in content:
                    found.append((str(filepath), term))
                    print(f"✅ ENCONTRADO em: {filepath}")
                    # Extrair contexto
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if term in line:
                            print(f"   Linha {i+1}: {line[:100]}")
                    print()
    except:
        pass

if not found:
    print("❌ 'Morro do Ouro' / 'Morro do Oitão' / 'Moura brasil' não encontrado em nenhum arquivo")
    print("\n💡 Talvez o usuário tenha um exemplo que não está catalogado?")
    print("   Ou existe um arquivo separado que não foi sincronizado?")

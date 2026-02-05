"""
Identificar e resolver conflitos de atribuição de facções
"""

import json

# Ver quais bairros conflitaram
with open('outputs/disputa_territory_from_kml.json', encoding='utf-8') as f:
    disputa = json.load(f)
    
with open('outputs/tcp_territory_from_kml.json', encoding='utf-8') as f:
    tcp = json.load(f)

disputa_names = set(a['name'] for a in disputa['assignments'])
tcp_names = set(a['name'] for a in tcp['assignments'])

conflitos = disputa_names & tcp_names

print('\nCONFLITOS DISPUTA vs TCP:')
print(f'Bairros em ambas as facções: {len(conflitos)}')
for name in sorted(conflitos):
    print(f'  {name}')

# Verificar DISPUTA que não conflitam
disputa_only = disputa_names - tcp_names
print(f'\nBairros APENAS em DISPUTA (sem conflito): {len(disputa_only)}')
for name in sorted(disputa_only):
    print(f'  {name}')

print(f'\n=> Sugestão: Usar prioridade TCP em conflitos (TCP é mais consolidado)')
print(f'   Os {len(conflitos)} bairros conflitantes ficarão com TCP')

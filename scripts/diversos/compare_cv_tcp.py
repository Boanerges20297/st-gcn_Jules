"""
Comparação entre CV e TCP
"""

import json
from pathlib import Path

BASE_DIR = Path('.')

print('\n' + '='*70)
print('📊 COMPARAÇÃO: CV vs TCP')
print('='*70)

# Carregar resultados
with open('outputs/tcp_validation.json', encoding='utf-8') as f:
    tcp_val = json.load(f)

print('\n🔴 COMANDO VERMELHO (CV):')
print('   Total atribuído: 150/319 (47.0%)')
print('   - Dentro polígono: 41')
print('   - Perto borda: 109')
print('   - Fortaleza: 136')
print('   - RMF: 6')
print('   - Interior: 8')

print('\n🔵 TCP - TERCEIRO COMANDO PURO:')
print(f'   Total atribuído: {tcp_val["total_assigned"]}/319 (13.5%)')
print(f'   - Dentro polígono: {tcp_val["inside_polygon"]}')
print(f'   - Perto borda: {tcp_val["near_boundary"]}')
print(f'   - Bairros Fortaleza: {tcp_val["bairros_fortaleza"]}')
print(f'   - Bairros RMF: {tcp_val["bairros_rmf"]}')
print(f'   - Cidades: {tcp_val["cidades"]}')

print('\n📍 COORDENADAS TERRITORIAIS:')
print('   CV:  Longitude -38.66 a -38.40, Latitude -3.90 a -3.65')
print('   TCP: Longitude -41.28 a -37.31, Latitude -7.69 a -2.89')
print('   (TCP cobre área MUITO MAIOR)')

print('\n📌 BAIRROS CV (41 DENTRO POLÍGONO):')
cv_bairros = ['AEROLÂNDIA', 'ALTO DA BALANÇA', 'ANCURI', 'AUTRAN NUNES', 'BELA VISTA', 
              'CARLITO PAMPLONA', 'CIDADE 2000', 'CONJUNTO CEARÁ I', 'CONJUNTO CEARÁ II', 
              'CRISTO REDENTOR', 'DIAS MACÊDO', 'ELLERY', 'GENIBAÚ', 'GRANJA LISBOA', 
              'GRANJA PORTUGAL', 'GUADALAJARA', 'HORIZONTE', 'INDUSTRIAL', 'IPARANA', 
              'JANGURUSSU', 'JARDIM DAS OLIVEIRAS', 'JARDIM IRACEMA', 'JOSÉ BONIFÁCIO', 
              'MARACANAÚ', 'MARECHAL RONDON', 'MONTE CASTELO', 'NOVO MONDUBIM', 'OLAVO OLIVEIRA', 
              'PARQUE ALBANO', 'PARQUE DAS NAÇÕES', 'PARQUE PRESIDENTE VARGAS', 'PARQUE SANTA ROSA', 
              'PARQUE SÃO JOSÉ', 'PAUPINA', 'PIRAMBU', 'PLANALTO AYRTON SENNA', 'PRAIA DO FUTURO II', 
              'SAPIRANGA/COITÉ', 'SÃO MIGUEL', 'VICENTE PINZÓN', 'VILA VELHA']
for b in sorted(cv_bairros):
    print(f'   ✓ {b}')

print('\n📌 BAIRROS TCP (10 DENTRO POLÍGONO):')
tcp_inside = tcp_val['assignments']
tcp_inside_only = [a['name'] for a in tcp_inside if a['in_polygon']]
for b in sorted(tcp_inside_only):
    print(f'   ✓ {b}')

print('\n✅ VALIDAÇÃO TCP CONCLUÍDA')
print('='*70)

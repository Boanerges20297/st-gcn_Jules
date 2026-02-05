#!/usr/bin/env python3
"""
Verificar quais tipos de eventos CVP existem nos dados brutos
"""

import json
from collections import Counter

print("\n" + "="*80)
print("ANÁLISE DE TIPOS DE CVP NOS DADOS BRUTOS")
print("="*80 + "\n")

# Carregar dados brutos
with open('data/raw/dados_status_ocorrencias_gerais.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Contar tipos de eventos
tipos = []
cvp_keywords = ['ROUBO', 'FURTO', 'VEÍCULO', 'VEICULO', 'MOTO', 'CARRO']

print("📊 AMOSTRA DE TIPOS DE EVENTOS CVP:\n")

# Pegar últimos 1000 eventos
recent_events = data[-10000:] if len(data) > 10000 else data

for event in recent_events:
    tipo = event.get('tipo_evento', '').upper()
    if any(kw in tipo for kw in cvp_keywords):
        tipos.append(tipo)

# Contar frequências
tipo_counts = Counter(tipos)

print(f"Total de eventos CVP encontrados: {len(tipos)}\n")
print("Top 30 tipos mais comuns:")
for tipo, count in tipo_counts.most_common(30):
    print(f"  {count:5d}x - {tipo}")

print("\n" + "="*80)
print("ANÁLISE DO FILTRO ATUAL")
print("="*80 + "\n")

# Simular filtro atual
veiculos_count = 0
outros_count = 0

for tipo in tipos:
    is_veiculo = any(kw in tipo for kw in ['VEÍCULO', 'VEICULO', 'MOTO', 'CARRO', 'AUTOMÓVEL'])
    is_roubo_furto = any(kw in tipo for kw in ['ROUBO', 'FURTO'])
    
    if is_veiculo and is_roubo_furto:
        veiculos_count += 1
    else:
        outros_count += 1

total = veiculos_count + outros_count
print(f"✅ Passam pelo filtro (VEÍCULO + ROUBO/FURTO): {veiculos_count} ({veiculos_count/total*100:.1f}%)")
print(f"❌ Bloqueados pelo filtro: {outros_count} ({outros_count/total*100:.1f}%)")
print()

if veiculos_count < total * 0.3:
    print("⚠️  FILTRO MUITO RESTRITIVO!")
    print("   Sugestão: Incluir 'MOTO' sem exigir palavra 'VEÍCULO'")

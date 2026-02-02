#!/usr/bin/env python3
"""
Análise detalhada de amplificação de eventos exógenos.

Mostra:
- Quais eventos estão sendo amplificados
- Severidade de cada evento
- Fator de amplificação aplicado
- Nós afetados
- Impacto na matriz de adjacência
"""

import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXOGENOUS_FILE = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'exogenous_events_cache.json')

# Mapa de amplificação (do código app.py)
AMPLIFICATION_MAP = {
    'HIGH': 1.2,    # Ultraleve
    'MEDIUM': 1.1,  # Ultraleve
    'LOW': 1.05     # Ultraleve
}

print("=" * 90)
print("ANÁLISE DETALHADA DE AMPLIFICAÇÃO DE EVENTOS EXÓGENOS")
print("=" * 90)

# Carregar eventos
if os.path.exists(EXOGENOUS_FILE):
    with open(EXOGENOUS_FILE, 'r', encoding='utf-8') as f:
        exogenous_events = json.load(f)
    
    print(f"\nEventos carregados: {len(exogenous_events)} lotes")
    
    # Analisar cada lote
    total_points = 0
    severity_count = defaultdict(int)
    amplification_total = 0.0
    point_details = []
    
    for batch_idx, batch in enumerate(exogenous_events):
        points = batch.get('points', [])
        
        for pt in points:
            if isinstance(pt, dict):
                severity = pt.get('conflict_severity', 'LOW')
            else:
                severity = 'LOW'
            
            if severity not in AMPLIFICATION_MAP:
                severity = 'LOW'
            
            total_points += 1
            severity_count[severity] += 1
            
            amp_factor = AMPLIFICATION_MAP[severity]
            amplification_total += amp_factor
            
            # Guardar detalhes
            desc = pt.get('description', 'N/A') if isinstance(pt, dict) else 'N/A'
            point_details.append({
                'batch': batch_idx + 1,
                'severity': severity,
                'factor': amp_factor,
                'description': desc[:80] if desc else 'N/A'
            })
    
    # Mostrar detalhes por severidade
    print(f"\nEventos por severidade:")
    for severity in ['HIGH', 'MEDIUM', 'LOW']:
        count = severity_count.get(severity, 0)
        factor = AMPLIFICATION_MAP.get(severity, 0)
        if count > 0:
            print(f"\n[{severity}] {count} eventos × {factor}x = {count * factor:.0f} unidades")
            # Mostrar exemplos
            examples = [p for p in point_details if p['severity'] == severity][:3]
            for ex in examples:
                print(f"  - Lote {ex['batch']}: {ex['description']}")
    
    if severity_count:
        print(f"\n" + "-" * 90)
    
    print(f"\n" + "=" * 90)
    print("RESUMO DE AMPLIFICAÇÃO")
    print("=" * 90)
    print(f"\nBreakdown por severidade:")
    for severity, count in severity_count.items():
        amp_factor = AMPLIFICATION_MAP.get(severity, 0)
        print(f"  {severity}: {count} lotes")
    
    print(f"\nTotal de pontos: {total_points}")
    print(f"Total de influência amplificada: {amplification_total:.0f} unidades")
    
    # Verificar cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        
        print(f"\n[CACHE STATUS]")
        print(f"  Amplificado: {cache['amplified']}")
        print(f"  Eventos: {cache['event_count']}")
        print(f"  Hash: {cache['hash'][:16]}...")
        
        if cache['amplified']:
            print(f"\n✓ Amplificação já foi APLICADA")
            print(f"  Próximos reloads: PULAM reamplificação (mesmos {total_points} pontos)")
        else:
            print(f"\n⚠ Amplificação NÃO foi aplicada ainda")
    
    print(f"\n" + "=" * 90)
    print("NOTAS TÉCNICAS")
    print("=" * 90)
    print(f"""
A amplificação funciona assim:
1. Para cada evento de severidade HIGH/MEDIUM/LOW
2. Busca nós vizinhos dentro de 500m (raio padrão)
3. Multiplica adjacência: adj_matrix[idx, :] *= fator
                          adj_matrix[:, idx] *= fator
4. Cache evita reamplificar no reload (mesmos eventos = skip)

Impacto na rede:
- adj_matrix é 319×319 (nodes × nodes)
- Amplificação afeta linhas/colunas de cada nó crítico
- Cascata: se node A amplificado, propaga influência para neighbors
- Resultado: 24 nós críticos originais + derivados = até 119+
""")
    
else:
    print(f"✗ Arquivo não encontrado: {EXOGENOUS_FILE}")

print("=" * 90)

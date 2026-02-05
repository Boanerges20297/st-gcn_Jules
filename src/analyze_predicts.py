#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Completa dos Arquivos de Predição
Valida consistência, correções do ranking e evolução temporal
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_predict_file(filepath):
    """Extrai informações estruturadas de um arquivo predict"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = {
        'filepath': filepath,
        'timestamp': None,
        'total_nodes': 0,
        'top_20_nodes': [],
        'demotions': [],
        'corrections_applied': False,
        'patterns': {}
    }
    
    # Extrai timestamp
    match = re.search(r'Timestamp: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
    if match:
        data['timestamp'] = match.group(1)
    
    # Extrai total de nodes
    match = re.search(r'Total de nodes analisados: (\d+)', content)
    if match:
        data['total_nodes'] = int(match.group(1))
    
    # Extrai top 20 nodes
    lines = content.split('\n')
    in_top20 = False
    for i, line in enumerate(lines):
        if 'RANKING ATUALIZADO (TOP 20 NODES)' in line:
            in_top20 = True
            continue
        if in_top20 and '---' in line:
            continue
        if in_top20 and line.strip() and not line.startswith('Rank'):
            if 'CORREÇÕES' in line or 'PADRÕES' in line or line.startswith('---'):
                in_top20 = False
                break
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rank = int(parts[0])
                    node = int(parts[1])
                    cvli_pct = parts[2].replace('%', '').strip()
                    data['top_20_nodes'].append({
                        'rank': rank,
                        'node': node,
                        'cvli_pct': cvli_pct
                    })
                except:
                    pass
    
    # Extrai correções (demotions)
    if 'Nenhuma demoção' in content or 'nenhuma demoção' in content.lower():
        data['corrections_applied'] = False
    else:
        data['corrections_applied'] = True
        # Procura por padrão de demoção
        demotion_match = re.findall(r'Node (\d+).*?demovido|demoção', content, re.IGNORECASE)
        data['demotions'] = demotion_match
    
    # Extrai padrões
    patterns = ['HISTÓRICO RECENTE', 'ALTA ATIVIDADE', 'EVENTOS EXÓGENOS', 'EVENTOS CRÍTICOS']
    for pattern in patterns:
        match = re.search(rf'{pattern}\s*\((\d+)\s*nodes\)', content)
        if match:
            data['patterns'][pattern] = int(match.group(1))
    
    return data

def main():
    predicts_dir = Path('predicts')
    
    if not predicts_dir.exists():
        print("❌ Pasta /predicts não encontrada!")
        return
    
    # Carrega todos os arquivos
    files = sorted(predicts_dir.glob('predict_*.txt'))
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISE DOS ARQUIVOS DE PREDIÇÃO")
    print(f"{'='*80}")
    print(f"✅ Encontrados {len(files)} arquivos")
    
    results = []
    for filepath in files:
        data = parse_predict_file(filepath)
        results.append(data)
    
    # Análise Temporal
    print(f"\n{'='*80}")
    print(f"⏱️  ANÁLISE TEMPORAL")
    print(f"{'='*80}")
    print(f"{'Arquivo':<30} {'Timestamp':<20} {'Nodes':<10} {'Top Node':<12} {'Correções':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        timestamp = r['timestamp'] or 'N/A'
        top_node = r['top_20_nodes'][0]['node'] if r['top_20_nodes'] else 'N/A'
        corrections = '✅ SIM' if r['corrections_applied'] else '❌ NÃO'
        filename = Path(r['filepath']).name
        print(f"{filename:<30} {timestamp:<20} {r['total_nodes']:<10} {str(top_node):<12} {corrections:<12}")
    
    # Análise de Consistência
    print(f"\n{'='*80}")
    print(f"🔍 CONSISTÊNCIA DOS RESULTADOS")
    print(f"{'='*80}")
    
    top_nodes_history = defaultdict(int)
    for r in results:
        for item in r['top_20_nodes'][:5]:  # Top 5
            top_nodes_history[item['node']] += 1
    
    print(f"\nNodes que aparecem no TOP-5 (histórico de predições):")
    sorted_nodes = sorted(top_nodes_history.items(), key=lambda x: x[1], reverse=True)
    for node, count in sorted_nodes[:10]:
        pct = (count / len(results)) * 100
        print(f"  Node {node:3d}: {count:2d} vezes ({pct:5.1f}%)")
    
    # Análise de Padrões
    print(f"\n{'='*80}")
    print(f"📈 PADRÕES IDENTIFICADOS")
    print(f"{'='*80}")
    
    pattern_stats = defaultdict(list)
    for r in results:
        for pattern, count in r['patterns'].items():
            pattern_stats[pattern].append(count)
    
    for pattern, counts in sorted(pattern_stats.items()):
        avg = sum(counts) / len(counts)
        min_val = min(counts)
        max_val = max(counts)
        print(f"\n{pattern}:")
        print(f"  Média: {avg:.1f} nodes")
        print(f"  Min-Max: {min_val}-{max_val}")
    
    # Análise de Correções
    print(f"\n{'='*80}")
    print(f"🔧 ANÁLISE DE CORREÇÕES DO RANKING")
    print(f"{'='*80}")
    
    with_corrections = sum(1 for r in results if r['corrections_applied'])
    without_corrections = len(results) - with_corrections
    pct_corrected = (with_corrections / len(results)) * 100 if results else 0
    
    print(f"Predições com correções: {with_corrections}/{len(results)} ({pct_corrected:.1f}%)")
    print(f"Predições sem correções: {without_corrections}/{len(results)} ({100-pct_corrected:.1f}%)")
    
    if with_corrections == 0:
        print(f"\n✅ RESULTADO: Nenhuma correção foi necessária!")
        print(f"   → ST-GCN está alinhado com ranking")
        print(f"   → Confiança alta nos scores originais")
    else:
        print(f"\n⚠️  {with_corrections} predições receberam correções")
    
    # Validação Geral
    print(f"\n{'='*80}")
    print(f"✅ VALIDAÇÃO GERAL")
    print(f"{'='*80}")
    
    issues = []
    
    # Check 1: Consistência de nodes
    if not all(r['total_nodes'] == 319 for r in results):
        issues.append("❌ Número de nodes inconsistente")
    else:
        print("✅ Todas as predições analisam 319 nodes")
    
    # Check 2: Top 20 sempre preenchido
    if not all(len(r['top_20_nodes']) == 20 for r in results):
        issues.append("❌ Nem todas as predições têm top-20 completo")
    else:
        print("✅ Todos os arquivos têm top-20 nodes completo")
    
    # Check 3: Timestamps válidos
    if not all(r['timestamp'] for r in results):
        issues.append("❌ Alguns arquivos sem timestamp válido")
    else:
        print("✅ Todos os timestamps são válidos")
    
    # Check 4: Padrões consistentes
    expected_patterns = {'HISTÓRICO RECENTE', 'ALTA ATIVIDADE', 'EVENTOS EXÓGENOS', 'EVENTOS CRÍTICOS'}
    for r in results:
        if set(r['patterns'].keys()) != expected_patterns:
            issues.append(f"❌ Padrões inconsistentes em {Path(r['filepath']).name}")
            break
    else:
        print("✅ Padrões consistentes em todas as predições")
    
    # Check 5: Ranking confiável
    if without_corrections / len(results) > 0.8:
        print("✅ Ranking confiável: >80% das predições não precisam correção")
    else:
        print("⚠️  Ranking com correções frequentes (ajustes contínuos)")
    
    if issues:
        print("\n⚠️  ISSUES DETECTADAS:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ NENHUM PROBLEMA DETECTADO - SISTEMA VALIDADO!")
    
    # Resumo Final
    print(f"\n{'='*80}")
    print(f"📋 RESUMO FINAL")
    print(f"{'='*80}")
    print(f"Total de predições: {len(results)}")
    print(f"Período: {results[0]['timestamp']} → {results[-1]['timestamp']}")
    print(f"Top node mais frequente: {sorted_nodes[0][0]} ({sorted_nodes[0][1]}/{len(results)} vezes)")
    print(f"Correções necessárias: {with_corrections}/{len(results)} ({pct_corrected:.1f}%)")
    print(f"Status: {'✅ VALIDADO E OPERACIONAL' if not issues else '⚠️  REVISAR ISSUES'}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

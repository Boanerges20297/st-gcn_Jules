"""
Script para adicionar conflict_severity a todos os eventos exógenos existentes
"""

import os
import json


def classify_conflict_severity(event):
    """Classifica severidade do conflito baseado no texto do evento"""
    
    # Coleta todos os campos de texto do point e do raw_event
    text_fields = []
    
    # Campos do próprio point
    for key in ['description']:
        value = event.get(key, '')
        if value:
            text_fields.append(str(value).lower())
    
    # Campos do raw_event (se existir)
    if 'raw_event' in event:
        raw = event['raw_event']
        for key in ['natureza', 'descricao', 'resumo', 'raw', 'raw_text']:
            value = raw.get(key, '')
            if value:
                text_fields.append(str(value).lower())
    
    txt = ' '.join(text_fields)
    
    # HIGH severity - sinais de execução/confronto
    high_keywords = [
        'amarrado', 'amarrados', 'amarradas', 'mãos amarradas', 'maos amarradas',
        'pés amarrados', 'pes amarrados', 'pernas amarradas', 'membros amarrados',
        'membros inferiores amarrados', 'membros superiores amarrados',
        'tortura', 'torturado', 'execução', 'executado', 'execucao',
        'carbonizado', 'queimado vivo', 'enterrado', 'sepultado',
        'duplo homicídio', 'duplo homicidio', 'triplo homicídio', 'triplo homicidio',
        'chacina', 'massacre', 'emboscada', 'tocaia',
        'disputa territorial', 'guerra de facções', 'guerra de faccoes',
        'decapitado', 'esquartejado', 'mutilado',
        'sinais de execução', 'sinais de execucao'
    ]
    
    # MEDIUM severity - violência armada
    medium_keywords = [
        'homicídio', 'homicidio', 
        'tiro', 'bala', 'disparos', 'disparo', 'tiros',
        'lesão a bala', 'lesao a bala', 'ferimento a bala',
        'fuzil', 'fuzilamento', 'metralhadora',
        'confronto', 'tiroteio', 'troca de tiros',
        'assassinato', 'assassinado',
        'facada', 'esfaqueado'
    ]
    
    # Verifica severidade (verifica HIGH primeiro)
    if any(keyword in txt for keyword in high_keywords):
        return 'HIGH'
    elif any(keyword in txt for keyword in medium_keywords):
        return 'MEDIUM'
    else:
        return 'LOW'


def add_conflict_severity_to_events():
    """Adiciona conflict_severity a todos os eventos exógenos"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    events_file = os.path.join(project_root, 'data', 'exogenous_events.json')
    
    print("="*80)
    print("ADICIONANDO CONFLICT_SEVERITY AOS EVENTOS EXÓGENOS")
    print("="*80)
    
    # Carrega eventos existentes
    if not os.path.exists(events_file):
        print(f"❌ Arquivo não encontrado: {events_file}")
        return
    
    with open(events_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    print(f"\n✓ Carregados {len(events)} lotes de eventos")
    
    # Estatísticas
    total_points = 0
    high_count = 0
    medium_count = 0
    low_count = 0
    updated_count = 0
    
    # Processa cada lote
    for batch_idx, batch in enumerate(events):
        points = batch.get('points', [])
        
        for point_idx, point in enumerate(points):
            total_points += 1
            
            # Se já tem conflict_severity, pula (a menos que seja recalcular)
            existing_severity = point.get('conflict_severity')
            
            # Classifica
            severity = classify_conflict_severity(point)
            
            # Atualiza
            if existing_severity != severity:
                point['conflict_severity'] = severity
                updated_count += 1
            elif not existing_severity:
                point['conflict_severity'] = severity
                updated_count += 1
            
            # Conta
            if severity == 'HIGH':
                high_count += 1
            elif severity == 'MEDIUM':
                medium_count += 1
            else:
                low_count += 1
    
    print(f"\n📊 Estatísticas:")
    print(f"  Total de eventos: {total_points}")
    print(f"  HIGH severity:    {high_count} ({high_count/total_points*100:.1f}%)")
    print(f"  MEDIUM severity:  {medium_count} ({medium_count/total_points*100:.1f}%)")
    print(f"  LOW severity:     {low_count} ({low_count/total_points*100:.1f}%)")
    print(f"  Atualizados:      {updated_count}")
    
    # Salva arquivo atualizado
    backup_file = events_file + '.backup'
    
    # Faz backup
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Backup salvo em: {backup_file}")
    
    # Salva arquivo atualizado
    with open(events_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Arquivo atualizado: {events_file}")
    
    # Mostra alguns exemplos
    print(f"\n📋 Exemplos de eventos HIGH severity:")
    high_examples = []
    for batch in events:
        for point in batch.get('points', []):
            if point.get('conflict_severity') == 'HIGH':
                desc = point.get('description', point.get('resumo', 'SEM DESCRIÇÃO'))[:80]
                high_examples.append(desc)
                if len(high_examples) >= 5:
                    break
        if len(high_examples) >= 5:
            break
    
    for i, example in enumerate(high_examples, 1):
        print(f"  {i}. {example}")
    
    print("\n" + "="*80)
    print("CONCLUÍDO")
    print("="*80)
    print(f"✅ {total_points} eventos processados")
    print(f"✅ {updated_count} eventos atualizados")
    print(f"✅ {high_count} eventos de alta severidade detectados")


if __name__ == '__main__':
    add_conflict_severity_to_events()

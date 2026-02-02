"""
Script para atualizar a severidade de conflito em eventos exógenos existentes.
Detecta casos de deslocamento forçado/expulsão e reclassifica como MEDIUM.
"""

import json
import os

def detect_displacement(text_fields):
    """Detecta sinais de deslocamento forçado ou expulsão."""
    txt = ' '.join(text_fields).lower()
    
    displacement_keywords = [
        'ameaças de grupo criminoso',
        'ameaça de grupo criminoso', 
        'expulsão',
        'expulsao',
        'deslocamento forçado',
        'deslocamento forcado',
        'precisa fazer a mudança',
        'precisa fazer mudanca',
        'forçado a sair',
        'forcado a sair',
        'obrigado a sair',
        'ameaças e precisa',
        'sofrendo ameaças'
    ]
    
    return any(k in txt for k in displacement_keywords)


def update_severity(events_file):
    """Atualiza a severidade dos eventos exógenos."""
    
    if not os.path.exists(events_file):
        print(f"Arquivo não encontrado: {events_file}")
        return
    
    with open(events_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    updated_count = 0
    
    for batch in events:
        points = batch.get('points', [])
        
        for pt in points:
            raw_event = pt.get('raw_event')
            if not raw_event:
                continue
            
            # Coleta campos de texto para análise
            text_fields = []
            for field in ['natureza', 'resumo', 'descricao', 'raw_text', 'description']:
                value = raw_event.get(field)
                if value:
                    text_fields.append(str(value))
            
            # Verifica se é um caso de deslocamento forçado
            if detect_displacement(text_fields):
                old_severity = raw_event.get('conflict_severity', 'LOW')
                
                # Atualiza para MEDIUM se for LOW
                if old_severity == 'LOW':
                    raw_event['conflict_severity'] = 'MEDIUM'
                    pt['conflict_severity'] = 'MEDIUM'
                    updated_count += 1
                    print(f"✓ Atualizado: {raw_event.get('natureza', 'N/A')} - {raw_event.get('resumo', '')[:80]}")
    
    # Salva o arquivo atualizado
    with open(events_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Total de eventos atualizados: {updated_count}")
    print(f"Arquivo salvo: {events_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    events_file = os.path.join(base_dir, 'data', 'exogenous_events.json')
    
    print("Atualizando severidade de eventos exógenos...")
    print(f"Arquivo: {events_file}\n")
    
    update_severity(events_file)

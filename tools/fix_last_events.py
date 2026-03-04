
import json
import os
import re

def fix_last_entries():
    path = 'data/exogenous_events.json'
    if not os.path.exists(path):
        print("Arquivo não encontrado.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        events = json.load(f)

    if len(events) < 3:
        print("Menos de 3 eventos para corrigir.")
        return

    # Corrigir as 3 últimas entradas
    for i in range(-3, 0):
        ev = events[i]
        raw = ev.get('raw_text', '')
        
        # Tenta extrair data e hora do raw_text se o date atual estiver incompleto (terminando em 00:00:00 ou similar)
        # Ex: "28/02/2026 09:36"
        dt_match = re.search(r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})', raw)
        final_dt = ev.get('date', '')
        
        if dt_match:
            d, t = dt_match.groups()
            d_pts = d.split('/')
            final_dt = f"{d_pts[2]}-{d_pts[1]}-{d_pts[0]} {t}:00"
        
        # Reconstruir o dicionário para garantir que 'date' seja a última chave
        new_ev = {}
        for k, v in ev.items():
            if k != 'date':
                new_ev[k] = v
        new_ev['date'] = final_dt
        events[i] = new_ev

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    
    print("As 3 últimas entradas foram corrigidas e o campo 'date' foi movido para o final.")

if __name__ == "__main__":
    fix_last_entries()

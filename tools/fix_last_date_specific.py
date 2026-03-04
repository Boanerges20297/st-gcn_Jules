
import json
import os

def fix_last_date():
    path = 'data/exogenous_events.json'
    if not os.path.exists(path): return

    with open(path, 'r', encoding='utf-8') as f:
        events = json.load(f)

    if events:
        last = events[-1]
        # Aplica a data do cabeçalho informada pelo usuário (29/02/2026)
        # Combinando com o horário já existente no registro (09:15)
        new_date = "2026-02-29 09:15:00"
        
        # Reordena para garantir 'date' no final
        new_ev = {k: v for k, v in last.items() if k != 'date'}
        new_ev['date'] = new_date
        events[-1] = new_ev

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        print(f"Data corrigida para o último evento: {new_date}")

if __name__ == "__main__":
    fix_last_date()

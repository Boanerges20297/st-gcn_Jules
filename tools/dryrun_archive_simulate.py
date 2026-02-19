import os
import re
import json
from datetime import datetime, timedelta

ROOT = os.getcwd()
SRC = os.path.join(ROOT, 'data', 'exogenous_events.json')


def parse_event_date(e):
    event_date = None
    dval = e.get('date') or e.get('event_date')
    if isinstance(dval, str) and dval.strip():
        # skip time-only strings like '22:10'
        if re.match(r'^\d{2}:\d{2}$', dval.strip()):
            dval = None
        else:
            try:
                event_date = datetime.strptime(dval.strip()[:10], '%Y-%m-%d').date()
            except Exception:
                event_date = None

    if event_date is None:
        ing = e.get('ingested_at')
        try:
            if isinstance(ing, str) and ing.strip():
                try:
                    ing_dt = datetime.strptime(ing.strip(), '%Y-%m-%d %H:%M:%S')
                except Exception:
                    try:
                        ing_dt = datetime.fromisoformat(ing.strip())
                    except Exception:
                        ing_dt = None
                if ing_dt:
                    event_date = ing_dt.date()
        except Exception:
            event_date = None

    return event_date


def simulate():
    if not os.path.exists(SRC):
        print('Source file not found:', SRC)
        return

    with open(SRC, 'r', encoding='utf-8') as f:
        events = json.load(f) or []

    cutoff_date = (datetime.now() - timedelta(days=7)).date()

    old_events = []
    current_events = []

    for e in events:
        ev_date = parse_event_date(e)
        if ev_date is None:
            current_events.append(e)
            continue
        if ev_date < cutoff_date:
            old_events.append(e)
        else:
            current_events.append(e)

    print('Total events in source:', len(events))
    print('Would archive (older than', cutoff_date.isoformat() + '):', len(old_events))
    print('Would keep (<=7 days):', len(current_events))

    if old_events:
        print('\nSample archived event (first):')
        s = old_events[0]
        print('  raw_text:', s.get('raw_text') or s.get('descricao'))
        print('  date:', s.get('date') or s.get('ingested_at'))
        print('  natureza:', s.get('natureza'))
        print('  conflict_severity:', s.get('conflict_severity'))

    # Provide a small diff summary by natureza counts
    from collections import Counter
    cnt = Counter([ (e.get('natureza') or 'UNKNOWN') for e in old_events ])
    if cnt:
        print('\nCounts of archived by natureza:')
        for k,v in cnt.most_common():
            print(f'  {k}: {v}')

    # No files written — dry-run complete

if __name__ == '__main__':
    simulate()

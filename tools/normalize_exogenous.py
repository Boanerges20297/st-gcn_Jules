import json, os
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXO = os.path.join(BASE, 'data', 'exogenous_events.json')
BACKUP = EXO + '.bak'

if not os.path.exists(EXO):
    print('No exogenous file found at', EXO)
    exit(0)

with open(EXO, 'r', encoding='utf-8') as f:
    data = json.load(f)

if not isinstance(data, list):
    print('Unexpected format: root is not a list')
    exit(1)

print(f'Loaded {len(data)} entries')

# compute existing max id
max_id = 0
for item in data:
    try:
        if 'id' in item:
            iid = int(item['id'])
            if iid > max_id: max_id = iid
    except Exception:
        pass

changed = 0
for item in data:
    updated = False
    # If old-style 'text' + 'created_at' without id
    if 'id' not in item:
        max_id += 1
        item['id'] = str(max_id)
        updated = True
    # rename text -> original_text
    if 'original_text' not in item and 'text' in item:
        item['original_text'] = item.pop('text')
        # created_at -> timestamp
        if 'created_at' in item and 'timestamp' not in item:
            item['timestamp'] = item.pop('created_at')
        updated = True
    # normalize points
    pts = item.get('points', [])
    new_pts = []
    for p in pts:
        # If point is already in desired format, ensure type exists
        if 'raw_event' in p and 'description' in p:
            if 'type' not in p:
                p['type'] = 'exogenous'
            new_pts.append(p)
            continue
        # Otherwise build raw_event
        raw_event = p.copy()
        # remove lat/lng from raw_event
        raw_event.pop('lat', None)
        raw_event.pop('lng', None)
        desc = raw_event.get('resumo') or raw_event.get('description') or raw_event.get('natureza') or ''
        new_p = {
            'description': desc,
            'lat': p.get('lat'),
            'lng': p.get('lng'),
            'raw_event': raw_event,
            'type': 'exogenous'
        }
        new_pts.append(new_p)
        updated = True
    if pts and updated:
        item['points'] = new_pts
    if updated:
        changed += 1

if changed > 0:
    print(f'Changes to apply: {changed}, backing up to', BACKUP)
    os.replace(EXO, BACKUP)
    with open(EXO, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('Normalization complete.')
else:
    print('No changes required.')

import json, re, os
p = os.path.join('data','exogenous_events.json')
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

mismatch = []
count = 0
for item in data:
    count += 1
    raw = item.get('original_text','')
    parts = [s.strip() for s in re.split(r'\s*-\s*', raw) if s.strip()]
    if len(parts) > 2:
        tail = parts[2:]
    else:
        tail = parts
    if tail and re.search(r"\d{1,2}:\d{2}", tail[-1]):
        tail = tail[:-1]
    while tail and re.match(r'AIS\d+', tail[-1].upper()):
        tail = tail[:-1]
    candidate = ''
    if tail:
        candidate = re.sub(r"[()\.]", '', tail[-1]).strip()
    stored = ''
    pts = item.get('points') or []
    if pts:
        revent = pts[0].get('raw_event') or pts[0].get('raw') or {}
        if isinstance(revent, dict):
            stored = revent.get('municipio','')
    def norm(s):
        return re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ ]+","",(s or '')).strip().upper()
    cand_n = norm(candidate)
    stored_n = norm(stored)
    if cand_n and stored_n and cand_n != stored_n:
        mismatch.append({'id': item.get('id'), 'candidate': candidate, 'stored': stored, 'original_text': raw})
    if cand_n and stored_n == 'FORTALEZA' and cand_n != 'FORTALEZA':
        mismatch.append({'id': item.get('id'), 'candidate': candidate, 'stored': stored, 'original_text': raw})

print(f'Total events: {count}\nMismatches found: {len(mismatch)}')
for m in mismatch[:20]:
    print('\n---')
    print('id:', m['id'])
    print('candidate (from raw):', m['candidate'])
    print('stored municipio:', m['stored'])
    print('raw text:', m['original_text'].replace('\n',' '))

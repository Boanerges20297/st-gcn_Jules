import json
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'outputs'
FILES = [
    'top20_micro_nodes_capital.geojson',
    'top20_micro_nodes_rmf.geojson',
    'top20_micro_nodes_interior.geojson',
    'top20_micro_nodes.geojson'
]

FORT_CENTER = (-38.5267, -3.8100)  # lon, lat

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

summary = {}
for fname in FILES:
    p = OUT / fname
    if not p.exists():
        print(f'MISSING: {p}')
        continue
    data = json.loads(p.read_text(encoding='utf-8'))
    feats = data.get('features', [])
    print(f'{fname}: {len(feats)} features')
    if 'capital' in fname or 'rmf' in fname or 'interior' in fname:
        dists = []
        for f in feats:
            c = f.get('geometry', {}).get('coordinates', [None, None])
            if c[0] is None:
                continue
            d = haversine(c[0], c[1], FORT_CENTER[0], FORT_CENTER[1])
            dists.append(d)
        if dists:
            print(f'  distance to Fortaleza center: min={min(dists):.0f} m, max={max(dists):.0f} m')
        # sample first 3
        for f in feats[:3]:
            props = f.get('properties', {})
            coords = f.get('geometry', {}).get('coordinates')
            print(f"  sample: rank={props.get('rank')} name={props.get('name')[:30]} score={int(props.get('score',0))} coords={coords} region={props.get('region')}")
    else:
        # combined file: check regions distribution
        counts = {}
        for f in feats:
            r = f.get('properties', {}).get('region')
            counts[r] = counts.get(r, 0) + 1
        print('  combined region counts:', counts)
        summary = counts

# quick checks
ok = True
expected = {'capital':20,'rmf':20,'interior':20}
for k,v in expected.items():
    if summary.get(k,0) != v:
        print(f'CHECK FAIL: expected {v} for {k}, found {summary.get(k,0)}')
        ok = False
if ok:
    print('\nValidation OK: all regions have 20 features and combined file sums to 60')
else:
    print('\nValidation FAILED: see checks above')

"""
Fix MARECHAL RONDON entries in the top20 micro-nodes output files.
Move the wrongly classified entry from capital to rmf.
"""
import json, copy

def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def save(p, d):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

# ---- 1. Remove from capital --------------------------------------------------
cap = load('outputs/top20_micro_nodes_capital.geojson')
wrongly_placed = [f for f in cap['features'] if 'MARECHAL' in str(f['properties'].get('name', '')).upper()]
cap['features'] = [f for f in cap['features'] if f not in wrongly_placed]

# Re-rank capital
for i, f in enumerate(cap['features'], 1):
    f['properties']['rank'] = i

save('outputs/top20_micro_nodes_capital.geojson', cap)
print(f"Capital: removed {len(wrongly_placed)} entry(ies). Now {len(cap['features'])} features.")

# ---- 2. Add to RMF (if not already present with correct data) ----------------
rmf = load('outputs/top20_micro_nodes_rmf.geojson')

for feat in wrongly_placed:
    feat = copy.deepcopy(feat)
    feat['properties']['municipality'] = 'Caucaia'
    feat['properties']['region'] = 'rmf'
    # Assign next rank
    feat['properties']['rank'] = len(rmf['features']) + 1
    rmf['features'].append(feat)
    print(f"RMF: added '{feat['properties']['micronodo']}' (rank {feat['properties']['rank']})")

save('outputs/top20_micro_nodes_rmf.geojson', rmf)

# ---- 3. Fix in combined file -------------------------------------------------
combo = load('outputs/top20_micro_nodes.geojson')
for feat in combo['features']:
    if 'MARECHAL' in str(feat['properties'].get('name', '')).upper():
        if feat['properties'].get('municipality') == 'Fortaleza':
            feat['properties']['municipality'] = 'Caucaia'
            feat['properties']['region'] = 'rmf'
            print(f"Combined: fixed '{feat['properties']['micronodo']}' -> Caucaia/rmf")

save('outputs/top20_micro_nodes.geojson', combo)
print("Done.")

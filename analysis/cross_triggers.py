"""
Cross CVLI, faction and vehicle-theft data to build faction territories and detect homicide triggers.
Outputs:
 - outputs/faction_territories.geojson
 - outputs/cross_triggers_summary.csv
 - outputs/cross_triggers_report.json

Run: python analysis/cross_triggers.py
"""
import os
import json
import pickle
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Helper: monotone chain convex hull (returns list of (lon,lat))
def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull(points):
    pts = sorted(points)
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

# Load metadata
meta_path = 'data/processed/metadata_producao_v2.json'
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
bairro_names = meta.get('bairros_normalizados') or meta.get('bairros')
features_names = meta.get('tensor_enriquecido_features', [])

# Load node features and dates
node_features_path = 'data/processed/graph_data/node_features.npy'
dates_path = 'data/processed/graph_data/dates.pkl'
if not os.path.exists(node_features_path) or not os.path.exists(dates_path):
    raise SystemExit('Missing node_features or dates. Expected in data/processed/graph_data/')

node = np.load(node_features_path)
with open(dates_path, 'rb') as f:
    dates = pickle.load(f)
# normalize date strings
dates = [pd.to_datetime(d).date() for d in dates]

# Determine axes: find axis equal to len(bairro_names) and len(dates) and len(features)
n_bairros = len(bairro_names)
n_dates = len(dates)
n_features = len(features_names) if features_names else None

# possible shapes
shape = node.shape
if n_features and n_features in shape and n_bairros in shape and n_dates in shape:
    # find mapping
    ax_feat = shape.index(n_features)
    ax_dates = shape.index(n_dates)
    ax_bairros = shape.index(n_bairros)
    # reorder to (bairros, dates, features)
    node = np.transpose(node, [ax_bairros, ax_dates, ax_feat])
else:
    # heuristics: common orientations
    if node.ndim == 3:
        # try (b,d,f) or (b,f,d) or (d,b,f)
        if node.shape[0] == n_bairros and node.shape[1] == n_dates:
            pass
        elif node.shape[0] == n_bairros and node.shape[2] == n_dates:
            node = node.transpose(0,2,1)
        elif node.shape[1] == n_bairros and node.shape[0] == n_dates:
            node = node.transpose(1,0,2)
        else:
            # best effort: assume (n_nodes, n_dates, n_features)
            pass
    else:
        raise SystemExit('Unexpected node_features shape: %s' % (shape,))

# Extract CVLI feature (if available)
if features_names and 'CVLI' in features_names:
    cvli_idx = features_names.index('CVLI')
else:
    cvli_idx = 0

cvli = node[:, :, cvli_idx]  # shape (n_bairros, n_dates)

# Load faction analysis
fac_path = 'data/processed/analise_movimentacao_faccoes.json'
with open(fac_path, 'r', encoding='utf-8') as f:
    fac = json.load(f)

# Assign dominant faction per bairro (first if multiple, "CONTESTED" if many)
bairro_to_faction = {}
for b in bairro_names:
    info = fac.get(b) or fac.get(b.upper()) or fac.get(b.title())
    if not info:
        bairro_to_faction[b] = 'UNKNOWN'
    else:
        facs = info.get('facoes_envolvidas', [])
        if len(facs) == 1:
            bairro_to_faction[b] = facs[0]
        elif len(facs) == 0:
            bairro_to_faction[b] = 'UNKNOWN'
        else:
            bairro_to_faction[b] = 'CONTESTED'

# Load centroids
centroids_path = 'data/raw/bairros_centros_latlong.json'
with open(centroids_path, 'r', encoding='utf-8') as f:
    cents = json.load(f)

# Normalize centroids keys to match bairro_names
centroid_map = {}
for k,v in cents.items():
    key = k.strip().upper()
    centroid_map[key] = (v['long'], v['lat'])

# Build GeoJSON territories per faction
faction_points = defaultdict(list)
for i,b in enumerate(bairro_names):
    assigned = bairro_to_faction.get(b,'UNKNOWN')
    key = b.strip().upper()
    if key in centroid_map:
        faction_points[assigned].append(centroid_map[key])

geo_features = []
for fac_name, pts in faction_points.items():
    if len(pts) >= 3:
        hull = convex_hull(pts)
        geom = {
            'type': 'Polygon',
            'coordinates': [[ [x,y] for x,y in hull ]]  # lon,lat
        }
    elif len(pts) == 2:
        geom = {'type':'LineString','coordinates': [[x,y] for x,y in pts]}
    elif len(pts) == 1:
        geom = {'type':'Point','coordinates': [pts[0][0], pts[0][1]]}
    else:
        continue
    geo_features.append({'type':'Feature', 'properties':{'faction':fac_name,'n_points':len(pts)}, 'geometry':geom})

out_geo = {'type':'FeatureCollection', 'features': geo_features}
os.makedirs('outputs', exist_ok=True)
with open('outputs/faction_territories.geojson','w',encoding='utf-8') as f:
    json.dump(out_geo, f, ensure_ascii=False, indent=2)

# Load occurrences: try CSV first then JSON files
occ_df = None
csv_path = 'data/raw/View_Ocorrencias_Operacionais_Modelo.csv'
csv_path2 = 'data/raw/View_Ocorrencias_Operacionais_Modelo_NORMALIZADO.csv'
if os.path.exists(csv_path):
    occ_df = pd.read_csv(csv_path, dtype=str, low_memory=False)
elif os.path.exists(csv_path2):
    occ_df = pd.read_csv(csv_path2, dtype=str, low_memory=False)
else:
    # read ocurrence_*.json files and concat
    rows = []
    raw_dir = 'data/raw'
    for fn in os.listdir(raw_dir):
        if 'ocorrencia' in fn.lower() and fn.lower().endswith('.json'):
            p = os.path.join(raw_dir,fn)
            try:
                with open(p,'r',encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        rows.extend(data)
                    elif isinstance(data, dict):
                        # some files are dict of records
                        for item in data.values():
                            rows.append(item)
            except Exception:
                continue
    if rows:
        occ_df = pd.DataFrame(rows)

if occ_df is None:
    raise SystemExit('No occurrences data found')

# Ensure columns
for c in ['Natureza','Data','BairroOcor','DescriçãoOcor']:
    if c not in occ_df.columns:
        occ_df[c] = ''
occ_df['Data'] = pd.to_datetime(occ_df['Data'], errors='coerce').dt.date
occ_df['BairroOcor'] = occ_df['BairroOcor'].fillna('').str.strip()
occ_df['Descricao'] = occ_df['DescriçãoOcor'].fillna('').astype(str)
occ_df['Natureza'] = occ_df['Natureza'].fillna('').astype(str)

# Identify vehicle-theft candidate rows
veh_mask = (
    occ_df['Natureza'].str.contains('VEÍCULO|VEICULO|VEÍCULO LOCALIZADO|VEÍCULO LOCALIZADO|VEÍCULO LOCALIZADO', case=False, na=False)
    | (
        occ_df['Natureza'].str.contains('ROUBO|FURTO', case=False, na=False)
        & occ_df['Descricao'].str.contains('motoc|moto|motocicleta|carro|veícul|veicul|automóvel', case=False, na=False)
    )
)
veh_df = occ_df[veh_mask].copy()

# Prepare per-bairro per-date counts
counts = defaultdict(lambda: defaultdict(int))
for _,r in veh_df.iterrows():
    d = r['Data']
    b = r['BairroOcor'] or ''
    counts[b][d] += 1

# Build aligned series per bairro (align to dates list)
results = []
for i,b in enumerate(bairro_names):
    bkey = b
    bkey_norm = bkey.strip()
    # build vehicle counts series
    veh_series = np.array([counts.get(bkey_norm,{}).get(d,0) for d in dates], dtype=float)
    cvli_series = cvli[i].astype(float)
    # detect spikes (rolling 30 days)
    window = 30
    if len(veh_series) < window:
        roll_mean = np.mean(veh_series) if len(veh_series)>0 else 0.0
        roll_std = np.std(veh_series) if len(veh_series)>0 else 0.0
        spikes_idx = np.where(veh_series > roll_mean + 2*roll_std)[0]
    else:
        roll_mean = pd.Series(veh_series).rolling(window=window, min_periods=3).mean().to_numpy()
        roll_std = pd.Series(veh_series).rolling(window=window, min_periods=3).std().to_numpy()
        spikes_idx = np.where((veh_series > (roll_mean + 2*roll_std)) & ~np.isnan(roll_mean))[0]

    # for each spike check if cvli increases in next 7 days vs previous 14 days
    triggered = 0
    for t in spikes_idx:
        before_start = max(0, t-13)
        before_mean = np.mean(cvli_series[before_start:t+1]) if t+1>before_start else 0.0
        after_mean = np.mean(cvli_series[t+1:t+8]) if t+1 < len(cvli_series) else 0.0
        if after_mean > before_mean:
            triggered += 1
    trigger_score = float(triggered)/len(spikes_idx) if len(spikes_idx)>0 else 0.0

    # lagged correlation up to 14 days
    max_corr = 0.0
    best_lag = 0
    for lag in range(0,15):
        if lag==0:
            x = veh_series
            y = cvli_series
        else:
            x = veh_series[:-lag]
            y = cvli_series[lag:]
        if x.std()==0 or y.std()==0 or len(x)<3:
            continue
        corr = np.corrcoef(x,y)[0,1]
        if abs(corr) > abs(max_corr):
            max_corr = corr
            best_lag = lag

    results.append({
        'bairro': b,
        'assigned_faction': bairro_to_faction.get(b,'UNKNOWN'),
        'vehicle_theft_total': int(veh_series.sum()),
        'vehicle_spikes': int(len(spikes_idx)),
        'cvli_total': float(cvli_series.sum()),
        'trigger_score': trigger_score,
        'max_lagged_corr': float(max_corr),
        'best_lag_days': int(best_lag)
    })

# Write CSV
dfres = pd.DataFrame(results)
dfres.to_csv('outputs/cross_triggers_summary.csv', index=False)

# Aggregate per-faction
fac_agg = []
for fac_name, grp in dfres.groupby('assigned_faction'):
    fac_agg.append({
        'faction': fac_name,
        'n_bairros': int(len(grp)),
        'vehicle_theft_total': int(grp['vehicle_theft_total'].sum()),
        'mean_trigger_score': float(grp['trigger_score'].mean()),
        'mean_max_corr': float(grp['max_lagged_corr'].mean())
    })
with open('outputs/cross_triggers_report.json','w',encoding='utf-8') as f:
    json.dump({'per_bairro': results, 'per_faction': fac_agg}, f, ensure_ascii=False, indent=2)

print('Wrote outputs/faction_territories.geojson, outputs/cross_triggers_summary.csv, outputs/cross_triggers_report.json')

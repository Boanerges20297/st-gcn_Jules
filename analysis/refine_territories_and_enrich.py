"""
Refine faction territories (concave hull if possible) and enrich time series with prisao/apreensao/exogenous events.
Outputs:
 - outputs/faction_territories_refined.geojson
 - outputs/enriched_timeseries_by_faction.csv
 - outputs/granger_results.json

Run: python analysis/refine_territories_and_enrich.py
"""
import os
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# Try shapely for concave hull (alpha-shape) fallback to convex hull
try:
    from shapely.geometry import Point, MultiPoint, mapping
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

# statsmodels for Granger test
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATS = True
except Exception:
    HAS_STATS = False

# Helpers
def convex_hull(points):
    pts = sorted(points)
    if len(pts) <= 1:
        return pts
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

# Load resources
meta_path = 'data/processed/metadata_producao_v2.json'
with open(meta_path,'r',encoding='utf-8') as f:
    meta = json.load(f)
bairro_names = meta.get('bairros_normalizados')

fac_path = 'data/processed/analise_movimentacao_faccoes.json'
with open(fac_path,'r',encoding='utf-8') as f:
    fac = json.load(f)

centroids_path = 'data/raw/bairros_centros_latlong.json'
with open(centroids_path,'r',encoding='utf-8') as f:
    cents = json.load(f)
centroid_map = {k.strip().upper(): (v['long'], v['lat']) for k,v in cents.items()}

# assign faction per bairro
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

# build points per faction
from collections import defaultdict
faction_points = defaultdict(list)
for b in bairro_names:
    f = bairro_to_faction.get(b,'UNKNOWN')
    key = b.strip().upper()
    if key in centroid_map:
        faction_points[f].append(centroid_map[key])

# build geojson features
features = []
for fac_name, pts in faction_points.items():
    if len(pts) == 0:
        continue
    geom = None
    if HAS_SHAPELY and len(pts) >= 3:
        mp = MultiPoint([Point(x,y) for x,y in pts])
        poly = mp.convex_hull
        geom = mapping(poly)
    else:
        if len(pts) >= 3:
            hull = convex_hull(pts)
            geom = {'type':'Polygon','coordinates':[[ [x,y] for x,y in hull ]]}
        elif len(pts) == 2:
            geom = {'type':'LineString','coordinates': [[x,y] for x,y in pts]}
        else:
            geom = {'type':'Point','coordinates': [pts[0][0], pts[0][1]]}
    features.append({'type':'Feature','properties':{'faction':fac_name,'n_points':len(pts)},'geometry':geom})

out = {'type':'FeatureCollection','features':features}
os.makedirs('outputs',exist_ok=True)
with open('outputs/faction_territories_refined.geojson','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)

# Now enrich time series: vehicle thefts, CVLI, prisao, apreensao, exogenous events
# Load occurrences
occ_df = None
csv_path = 'data/raw/View_Ocorrencias_Operacionais_Modelo.csv'
csv_path2 = 'data/raw/View_Ocorrencias_Operacionais_Modelo_NORMALIZADO.csv'
if os.path.exists(csv_path):
    occ_df = pd.read_csv(csv_path, dtype=str, low_memory=False)
elif os.path.exists(csv_path2):
    occ_df = pd.read_csv(csv_path2, dtype=str, low_memory=False)
else:
    # try other json occurrence files
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
                        for item in data.values():
                            rows.append(item)
            except Exception:
                continue
    if rows:
        occ_df = pd.DataFrame(rows)

if occ_df is None:
    raise SystemExit('No occurrences data found')

for c in ['Natureza','Data','BairroOcor','DescriçãoOcor']:
    if c not in occ_df.columns:
        occ_df[c] = ''
occ_df['Data'] = pd.to_datetime(occ_df['Data'], errors='coerce').dt.date
occ_df['BairroOcor'] = occ_df['BairroOcor'].fillna('').str.strip()
occ_df['Descricao'] = occ_df['DescriçãoOcor'].fillna('').astype(str)
occ_df['Natureza'] = occ_df['Natureza'].fillna('').astype(str)

# classify rows
veh_mask = (
    occ_df['Natureza'].str.contains('VEÍCULO|VEICULO|VEÍCULO LOCALIZADO|VEÍCULO LOCALIZADO', case=False, na=False)
    | (
        occ_df['Natureza'].str.contains('ROUBO|FURTO', case=False, na=False)
        & occ_df['Descricao'].str.contains('motoc|moto|motocicleta|carro|veícul|veicul|automóvel', case=False, na=False)
    )
)
prisao_mask = occ_df['Natureza'].str.contains('PRIS', case=False, na=False)
apreensao_mask = occ_df['Natureza'].str.contains('APREENS', case=False, na=False)
cvli_mask = occ_df['Natureza'].str.contains('HOMICID|CVLI', case=False, na=False)

veh_df = occ_df[veh_mask].copy()
prisao_df = occ_df[prisao_mask].copy()
apreensao_df = occ_df[apreensao_mask].copy()
cvli_df = occ_df[cvli_mask].copy()

# Build date index from occurrences
all_dates = sorted(occ_df['Data'].dropna().unique())
# aggregate per-bairro
def build_counts(df):
    counts = defaultdict(lambda: defaultdict(int))
    for _,r in df.iterrows():
        d = r['Data']
        b = r['BairroOcor'] or ''
        counts[b][d] += 1
    return counts

veh_counts = build_counts(veh_df)
prisao_counts = build_counts(prisao_df)
apreensao_counts = build_counts(apreensao_df)
cvli_counts = build_counts(cvli_df)

# per-bairro series aligned
bairros = bairro_names
rows = []
for b in bairros:
    bnorm = b.strip()
    veh_series = [veh_counts.get(bnorm,{}).get(d,0) for d in all_dates]
    prisao_series = [prisao_counts.get(bnorm,{}).get(d,0) for d in all_dates]
    apreensao_series = [apreensao_counts.get(bnorm,{}).get(d,0) for d in all_dates]
    cvli_series = [cvli_counts.get(bnorm,{}).get(d,0) for d in all_dates]
    rows.append({'bairro':b,'veh_total':sum(veh_series),'prisao_total':sum(prisao_series),'apreensao_total':sum(apreensao_series),'cvli_total':sum(cvli_series)})

df_summary = pd.DataFrame(rows)
df_summary.to_csv('outputs/enriched_timeseries_by_bairro.csv', index=False)

# Aggregate per-faction
fac_rows = []
for fac_name, grp in df_summary.groupby(df_summary['bairro'].map(lambda x: bairro_to_faction.get(x,'UNKNOWN'))):
    fac_rows.append({'faction': fac_name,
                     'veh_total': int(grp['veh_total'].sum()),
                     'prisao_total': int(grp['prisao_total'].sum()),
                     'apreensao_total': int(grp['apreensao_total'].sum()),
                     'cvli_total': int(grp['cvli_total'].sum()),
                     'n_bairros': int(len(grp))})

df_fac = pd.DataFrame(fac_rows)
# write
df_fac.to_csv('outputs/enriched_timeseries_by_faction.csv', index=False)

# Granger causality tests for top N factions and top N bairros
import math
results = {'per_faction':[], 'per_bairro':[]}
if HAS_STATS:
    maxlag = 7
    # per-faction: build time series (aligned)
    for fac_name in df_fac['faction'].tolist():
        # build series by summing bairro series
        idxs = [i for i,b in enumerate(bairros) if bairro_to_faction.get(b,'UNKNOWN')==fac_name]
        if not idxs:
            continue
        veh_ts = np.array([sum(veh_counts.get(b.strip(),{}).get(d,0) for b in [bairros[i] for i in idxs]) for d in all_dates], dtype=float)
        cvli_ts = np.array([sum(cvli_counts.get(b.strip(),{}).get(d,0) for b in [bairros[i] for i in idxs]) for d in all_dates], dtype=float)
        df = pd.DataFrame({'veh':veh_ts, 'cvli':cvli_ts})
        # need enough non-zero length
        if len(df) < maxlag+3 or df['veh'].std()==0 or df['cvli'].std()==0:
            results['per_faction'].append({'faction':fac_name,'granger':None})
            continue
        try:
            gc = grangercausalitytests(df[['cvli','veh']], maxlag=maxlag, verbose=False)
            pvals = {lag:float(gc[lag][0]['ssr_ftest'][1]) for lag in gc}
            best = min(pvals.items(), key=lambda x: x[1])
            results['per_faction'].append({'faction':fac_name,'best_pval':best[1],'best_lag':int(best[0])})
        except Exception as e:
            results['per_faction'].append({'faction':fac_name,'error':str(e)})
    # per-bairro top by veh_total
    topN = 20
    top_b = df_summary.sort_values('veh_total', ascending=False).head(topN)['bairro'].tolist()
    for b in top_b:
        veh_ts = np.array([veh_counts.get(b.strip(),{}).get(d,0) for d in all_dates], dtype=float)
        cvli_ts = np.array([cvli_counts.get(b.strip(),{}).get(d,0) for d in all_dates], dtype=float)
        df = pd.DataFrame({'veh':veh_ts, 'cvli':cvli_ts})
        if len(df) < maxlag+3 or df['veh'].std()==0 or df['cvli'].std()==0:
            results['per_bairro'].append({'bairro':b,'granger':None})
            continue
        try:
            gc = grangercausalitytests(df[['cvli','veh']], maxlag=maxlag, verbose=False)
            pvals = {lag:float(gc[lag][0]['ssr_ftest'][1]) for lag in gc}
            best = min(pvals.items(), key=lambda x: x[1])
            results['per_bairro'].append({'bairro':b,'best_pval':best[1],'best_lag':int(best[0])})
        except Exception as e:
            results['per_bairro'].append({'bairro':b,'error':str(e)})
else:
    results['note'] = 'statsmodels not available; install statsmodels to run Granger tests'

with open('outputs/granger_results.json','w',encoding='utf-8') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

print('Wrote outputs/faction_territories_refined.geojson, outputs/enriched_timeseries_by_bairro.csv, outputs/enriched_timeseries_by_faction.csv, outputs/granger_results.json')
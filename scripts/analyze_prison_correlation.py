"""
analyze_prison_correlation.py

Loads prison event records from data/raw/data_with_coordinates.js, maps events to graph nodes
using rounded centroids, constructs per-node time series of prison counts aligned to processed
graph dates, and computes Pearson correlations between prison counts and CVLI/CVP event counts
for several windows. Also computes train/test split comparisons to flag potential overfitting.

Outputs: scripts/prison_correlation.json

Usage: python scripts/analyze_prison_correlation.py
"""
import os
import sys
import json
import math
from datetime import datetime
import numpy as np
import unicodedata

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
RAW_JS = os.path.join(ROOT, 'data', 'raw', 'data_with_coordinates.js')
OUT = os.path.join('scripts', 'prison_correlation.json')

if not os.path.exists(DATA_FILE):
    raise RuntimeError(f'Processed data not found: {DATA_FILE}')
if not os.path.exists(RAW_JS):
    raise RuntimeError(f'Raw prison data not found: {RAW_JS}')

import pickle

def safe_load_processed(path):
    """Attempt to unpickle while avoiding import errors for geopandas/shapely by
    mapping unknown classes to plain dicts where safe."""
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except ModuleNotFoundError as e:
        missing = str(e)
        print('Module missing during unpickle:', missing)
        # Fallback: try to load chunked files from data/processed/graph_data
        graph_dir = os.path.join(os.path.dirname(path), 'graph_data')
        if os.path.isdir(graph_dir):
            print('Attempting to load graph_data chunks from', graph_dir)
            pack = {}
            nf = os.path.join(graph_dir, 'node_features.npy')
            ng = os.path.join(graph_dir, 'nodes_gdf.json')
            dates_json = os.path.join(graph_dir, 'dates.json')
            if os.path.exists(nf):
                pack['node_features'] = np.load(nf, allow_pickle=True)
            if os.path.exists(ng):
                try:
                    with open(ng, 'r', encoding='utf-8') as fh:
                        gj = json.load(fh)
                    # extract geometries list (expect GeoJSON FeatureCollection)
                    feats = gj.get('features', [])
                    # represent nodes_gdf as a lightweight list of dicts with 'geometry' key
                    nodes = []
                    for f in feats:
                        geom = f.get('geometry')
                        nodes.append({'geometry': geom, 'properties': f.get('properties', {})})
                    pack['nodes_gdf'] = nodes
                except Exception:
                    pass
            if os.path.exists(dates_json):
                try:
                    with open(dates_json, 'r', encoding='utf-8') as fh:
                        pack['dates'] = json.load(fh)
                except Exception:
                    pass
            # fallback to neighborhood_coordinates backup and node_feature_tensor backup
            if ('nodes_gdf' not in pack) or (pack.get('nodes_gdf') in (None, [])):
                coord_backups = [p for p in os.listdir(os.path.dirname(path)) if p.startswith('neighborhood_coordinates_backup') and p.endswith('.npy')]
                if coord_backups:
                    coord_np = os.path.join(os.path.dirname(path), coord_backups[-1])
                    try:
                        coords = np.load(coord_np)
                        # coords expected shape (N,2) as [lat,lon] or [lon,lat]
                        pack['nodes_gdf'] = []
                        for c in coords:
                            try:
                                lat = float(c[0]); lon = float(c[1])
                            except Exception:
                                lat = float(c[1]); lon = float(c[0])
                            pack['nodes_gdf'].append({'geometry': {'type':'Point','coordinates':[lon,lat]}, 'properties':{}})
                    except Exception:
                        pass
            if 'node_features' not in pack:
                nf_backups = [p for p in os.listdir(os.path.dirname(path)) if p.startswith('node_feature_tensor_backup') and p.endswith('.npy')]
                if nf_backups:
                    nfnp = os.path.join(os.path.dirname(path), nf_backups[-1])
                    try:
                        pack['node_features'] = np.load(nfnp, allow_pickle=True)
                    except Exception:
                        pass
            return pack
        raise

# Prefer direct backup tensors (avoid geopandas unpickle). If not present, fall back.
loaded_from_backups = False
proc_dir = os.path.abspath(os.path.join(os.getcwd(), 'data', 'processed'))
nf_backups = [p for p in os.listdir(proc_dir) if p.startswith('node_feature_tensor_backup') and p.endswith('.npy')]
coord_backups = [p for p in os.listdir(proc_dir) if p.startswith('neighborhood_coordinates_backup') and p.endswith('.npy')]
dates_json_path = os.path.join(proc_dir, 'dates.json')
dates_json_graph = os.path.join(proc_dir, 'graph_data', 'dates.json')
if nf_backups and coord_backups and (os.path.exists(dates_json_path) or os.path.exists(dates_json_graph)):
    try:
        node_features = np.load(os.path.join(proc_dir, nf_backups[-1]), allow_pickle=True)
        coords = np.load(os.path.join(proc_dir, coord_backups[-1]))
        # load dates: prefer dates.json if non-empty, else try dates.pkl (graph_data)
        def load_dates_from(path_json, path_pkl, path_graph_pkl):
            if os.path.exists(path_json) and os.path.getsize(path_json) > 10:
                try:
                    return json.load(open(path_json,'r',encoding='utf-8'))
                except Exception:
                    pass
            # try pkl
            if os.path.exists(path_pkl):
                try:
                    return pickle.load(open(path_pkl,'rb'))
                except Exception:
                    pass
            if os.path.exists(path_graph_pkl):
                try:
                    return pickle.load(open(path_graph_pkl,'rb'))
                except Exception:
                    pass
            return None

        dates = load_dates_from(dates_json_path, os.path.join(proc_dir,'dates.pkl'), os.path.join(proc_dir,'graph_data','dates.pkl'))
        nodes_gdf = []
        # coords may be stored as [lon, lat] or [lat, lon]; detect and normalize
        for c in coords:
            c0 = float(c[0]); c1 = float(c[1])
            # if absolute first coord > 90, it's likely longitude
            if abs(c0) > 90:
                lon = c0; lat = c1
            else:
                # if second coord > 90 it's lon in second pos
                if abs(c1) > 90:
                    lon = c1; lat = c0
                else:
                    # default assume (lon,lat) if lon in typical negative for this region
                    lon = c0; lat = c1
            nodes_gdf.append({'geometry': {'type':'Point','coordinates':[lon,lat]}, 'properties':{}})
        # node_features backup may have shape (T,N,F); transpose to (N,T,F) when needed
        try:
            if isinstance(node_features, np.ndarray) and node_features.ndim==3:
                t0,n0,f0 = node_features.shape
                if len(dates)==t0 and len(nodes_gdf)==n0:
                    node_features = np.transpose(node_features, (1,0,2))
        except Exception:
            pass
        loaded_from_backups = True
        print('Loaded node_features, coords and dates from data/processed backups')
    except Exception:
        loaded_from_backups = False

if not loaded_from_backups:
    # attempt to load processed pack or fall back to chunked graph_data
    pack = safe_load_processed(DATA_FILE)
    if isinstance(pack, dict):
        node_features = pack.get('node_features')
        nodes_gdf = pack.get('nodes_gdf')
        dates = pack.get('dates')

if node_features is None or nodes_gdf is None or dates is None:
    raise RuntimeError('Missing node_features / nodes_gdf / dates in processed pack or backups')

# ensure node_features has shape (N, T, F)
if isinstance(node_features, np.ndarray) and node_features.ndim==3:
    a,b,c = node_features.shape
    # if shape is (T,N,F) transpose to (N,T,F)
    if b == len(nodes_gdf) and a != len(nodes_gdf):
        node_features = np.transpose(node_features, (1,0,2))
N, T, F = node_features.shape
print(f'Loaded processed graph: N={N}, T={T}, F={F}, dates={len(dates)}')

# parse raw JS file heuristically: extract first JSON array-like substring
txt = open(RAW_JS, 'r', encoding='utf-8').read()
start = txt.find('[')
end = txt.rfind(']')
if start==-1 or end==-1:
    raise RuntimeError('Could not locate JSON array in raw JS file')
raw_json = txt[start:end+1]
try:
    records = json.loads(raw_json)
except Exception:
    # try to fix common trailing commas
    import re
    cleaned = re.sub(r',\s*\]', ']', raw_json)
    records = json.loads(cleaned)

print('Parsed', len(records), 'raw records')

# determine raw records date range (attempt several common keys)
rec_dates = []
for r in records:
    for k in ('Data','date','Registro','RegistroData','DataRegistro'):
        v = r.get(k)
        if v:
            try:
                rec_dates.append(str(v)[:10])
            except Exception:
                pass
            break
    else:
        v = r.get('Registro') or r.get('Data')
        if v:
            rec_dates.append(str(v)[:10])

if rec_dates:
    rec_dates_sorted = sorted(set(rec_dates))
    rec_start = rec_dates_sorted[0]
    rec_end = rec_dates_sorted[-1]
else:
    rec_start = None; rec_end = None

# normalize dates object (could be pandas Index)
def first_last_dates(dates_obj):
    try:
        if hasattr(dates_obj, '__len__') and len(dates_obj)>0:
            d0 = dates_obj[0]
            d1 = dates_obj[-1]
            return str(d0)[:10], str(d1)[:10]
    except Exception:
        pass
    return (None, None)

model_start, model_end = first_last_dates(dates)

# compute overlap
overlap_start = None
overlap_end = None
if rec_start and model_start:
    try:
        overlap_start = max(rec_start, model_start)
        overlap_end = min(rec_end, model_end)
    except Exception:
        overlap_start = None; overlap_end = None

if overlap_start is None or overlap_end is None or overlap_start > overlap_end:
    print('WARNING: No temporal overlap between prison raw data and model dates')
else:
    print(f'Raw prison data covers {rec_start} → {rec_end}; model covers {model_start} → {model_end}; overlap: {overlap_start} → {overlap_end}')

# build rounded centroids map (2 decimals)
centroids = []
if isinstance(nodes_gdf, list):
    for item in nodes_gdf:
        geom = item.get('geometry') if isinstance(item, dict) else None
        if not geom:
            centroids.append((None, None)); continue
        coords = geom.get('coordinates') if isinstance(geom, dict) else None
        if not coords or len(coords) < 2:
            centroids.append((None, None)); continue
        lon = float(coords[0]); lat = float(coords[1])
        centroids.append((round(lat,2), round(lon,2)))
else:
    for idx, row in nodes_gdf.reset_index(drop=True).iterrows():
        geom = row.get('geometry')
        if geom is None:
            centroids.append((None, None))
            continue
        c = geom.centroid
        lat = float(c.y); lon = float(c.x)
        centroids.append((round(lat,2), round(lon,2)))

coord_to_nodes = {}
for i,(lat,lon) in enumerate(centroids):
    if lat is None: continue
    key = f"{lat}_{lon}"
    coord_to_nodes.setdefault(key, []).append(i)

unique_coords = set(coord_to_nodes.keys())
print('Unique rounded node coords:', len(unique_coords))

# load static bairro coords as fallback mapping
static_bairros = {}
static_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'static', 'fortaleza_bairros_coords.json')
if os.path.exists(static_file):
    try:
        static_bairros = json.load(open(static_file, 'r', encoding='utf-8'))
    except Exception:
        static_bairros = {}

def normalize_name(s):
    if not s:
        return ''
    s2 = str(s).strip().lower()
    s2 = unicodedata.normalize('NFKD', s2)
    s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
    return s2

# build normalized static map
static_norm = {normalize_name(k): tuple(v) for k,v in static_bairros.items()} if static_bairros else {}

# build mapping from static bairro name -> nearest node index (by centroid distance)
static_to_node = {}
node_centroids_exact = []
if isinstance(nodes_gdf, list):
    for item in nodes_gdf:
        geom = item.get('geometry') if isinstance(item, dict) else None
        if not geom:
            node_centroids_exact.append((None,None)); continue
        ccoords = geom.get('coordinates')
        if not ccoords or len(ccoords)<2:
            node_centroids_exact.append((None,None)); continue
        lon = float(ccoords[0]); lat = float(ccoords[1])
        node_centroids_exact.append((lat, lon))
else:
    try:
        for idx, row in nodes_gdf.reset_index(drop=True).iterrows():
            c = row.get('geometry')
            if c is None:
                node_centroids_exact.append((None,None)); continue
            cent = c.centroid
            node_centroids_exact.append((float(cent.y), float(cent.x)))
    except Exception:
        node_centroids_exact = [(None,None)] * len(centroids)

def haversine_km(a,b):
    # a=(lat,lon), b=(lat,lon)
    import math
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2-lat1; dlon = lon2-lon1
    R = 6371.0
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

if static_norm and any(x[0] is not None for x in node_centroids_exact):
    for sname_norm, sc in static_norm.items():
        # sc is (lat, lon) from static file
        try:
            slat = float(sc[0]); slon = float(sc[1])
        except Exception:
            continue
        best_i = None; best_d = None
        for i,(nlat,nlon) in enumerate(node_centroids_exact):
            if nlat is None: continue
            try:
                d = haversine_km((slat, slon), (nlat, nlon))
            except Exception:
                continue
            if best_d is None or d < best_d:
                best_d = d; best_i = i
        if best_i is not None:
            static_to_node[sname_norm] = best_i

# helper to extract lat/lon and date from record
def extract_record(rec):
    lat = rec.get('lat') or rec.get('latitude') or rec.get('y') or rec.get('LAT') or rec.get('latitud')
    lon = rec.get('lng') or rec.get('lon') or rec.get('longitude') or rec.get('x') or rec.get('LON') or rec.get('long')
    dt = rec.get('date') or rec.get('datetime') or rec.get('data') or rec.get('date_ocorrencia') or rec.get('created_at')
    if lat is None or lon is None or dt is None:
        return None
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return None
    # normalize date to ISO YYYY-MM-DD
    dstr = str(dt)[:10]
    try:
        _ = datetime.fromisoformat(dstr)
    except Exception:
        # attempt other common formats
        for fmt in ('%d/%m/%Y','%Y/%m/%d','%d-%m-%Y','%Y-%m-%d'):
            try:
                d = datetime.strptime(dstr, fmt)
                dstr = d.date().isoformat()
                break
            except Exception:
                continue
        # also return neighborhood if present
        bairro = rec.get('BairroOcor') or rec.get('BairroAbord') or rec.get('bairro')
        return (round(lat,2), round(lon,2), dstr, bairro)

# build per-node daily counts aligned to dates
date_index = {str(d)[:10]: i for i,d in enumerate(dates)}
node_prison = np.zeros((N, len(dates)), dtype=int)
unknown_count = 0
for rec in records:
    ex = extract_record(rec)
    if ex is None:
        unknown_count += 1
        continue
    lat, lon, dstr, bairro = ex
    # Prefer mapping by neighborhood name (`BairroOcor`) rather than lat/lon
    found_nodes = None
    if bairro:
        bnorm = normalize_name(bairro)
        # exact normalized match to static names
        if bnorm in static_norm:
            node_idx = static_to_node.get(bnorm)
            if node_idx is not None:
                found_nodes = [node_idx]
        else:
            # fuzzy character-overlap match: shared characters / len(rec_chars) >= 0.5
            rec_chars = set(bnorm.replace(' ','').replace('-',''))
            best = (None, 0.0)
            for sname in static_norm.keys():
                s_chars = set(sname.replace(' ','').replace('-',''))
                if not rec_chars:
                    continue
                overlap = len(rec_chars & s_chars) / float(len(rec_chars))
                if overlap > best[1]:
                    best = (sname, overlap)
            if best[0] and best[1] >= 0.5:
                node_idx = static_to_node.get(best[0])
                if node_idx is not None:
                    found_nodes = [node_idx]
    # fallback to lat/lon-based mapping if bairro mapping failed
    if not found_nodes:
        key = f"{lat}_{lon}"
        found_nodes = coord_to_nodes.get(key)
    if not found_nodes:
        unknown_count += 1
        continue
    if dstr not in date_index:
        # event outside processed dates; skip
        continue
    di = date_index[dstr]
    for ni in found_nodes:
        node_prison[ni, di] += 1

print('Mapped records; unknown/skipped:', unknown_count)
num_nodes_with = int(np.sum(np.any(node_prison>0, axis=1)))
print('Nodes with prison events:', num_nodes_with, 'of', N)

# If mapping by coordinates/bairro failed (common when processed node coords backup is coarse),
# try using the preprocessed `prisoes_with_features.parquet` which contains `bairro_id` per record.
if num_nodes_with == 0:
    pfile = os.path.join(proc_dir, 'prisoes_with_features.parquet')
    try:
        import pandas as pd
        if os.path.exists(pfile):
            dfp = pd.read_parquet(pfile)
            # expect columns 'Data' and 'bairro_id'
            if 'Data' in dfp.columns and 'bairro_id' in dfp.columns:
                # normalize date strings
                dfp['date'] = pd.to_datetime(dfp['Data']).dt.date.astype(str)
                # count events per bairro_id x date
                grp = dfp.groupby(['bairro_id','date']).size().reset_index(name='count')
                date_idx = {str(d)[:10]: i for i,d in enumerate(dates)}
                for _, row in grp.iterrows():
                    bid = int(row['bairro_id'])
                    dstr = row['date']
                    if dstr not in date_idx: continue
                    di = date_idx[dstr]
                    if 0 <= bid < N:
                        node_prison[bid, di] = int(row['count'])
                num_nodes_with = int(np.sum(np.any(node_prison>0, axis=1)))
                print('Loaded prison counts from prisoes_with_features.parquet; nodes_with_prison_events=', num_nodes_with)
    except Exception as e:
        print('Failed to load prisoes_with_features.parquet fallback:', e)
    # If still zero, attempt Fortaleza-only bairro-name fuzzy mapping using static_bairros
    if num_nodes_with == 0 and static_norm and static_to_node:
                print('Attempting Fortaleza-only bairro name mapping (fuzzy >=50%)')
                for rec in records:
                    city = rec.get('CidadeOcor') or rec.get('cidade') or rec.get('Cidade')
                    if not city: continue
                    if normalize_name(city) != 'fortaleza':
                        continue
                    # get bairro
                    bairro = rec.get('BairroOcor') or rec.get('BairroAbord') or rec.get('bairro')
                    if not bairro: continue
                    bnorm = normalize_name(bairro)
                    rec_chars = set(bnorm.replace(' ','').replace('-',''))
                    if not rec_chars: continue
                    best = (None, 0.0)
                    for sname in static_norm.keys():
                        s_chars = set(sname.replace(' ','').replace('-',''))
                        overlap = len(rec_chars & s_chars) / float(len(rec_chars))
                        if overlap > best[1]:
                            best = (sname, overlap)
                    if best[0] and best[1] >= 0.5:
                        node_idx = static_to_node.get(best[0])
                        if node_idx is not None:
                            # date
                            dstr = str(rec.get('Data') or rec.get('date') or rec.get('Registro') )[:10]
                            if dstr in date_index:
                                di = date_index[dstr]
                                node_prison[node_idx, di] += 1
                num_nodes_with = int(np.sum(np.any(node_prison>0, axis=1)))
                print('After Fortaleza-only mapping, nodes_with_prison_events=', num_nodes_with)

WINDOWS = [30,60,90,180]
results = {}

for w in WINDOWS:
    if T < w:
        continue
    res_w = {}
    # event sums for CVLI (col 0) and CVP (col 1 if exists)
    ev_cvli = np.sum(node_features[:, -w:, 0], axis=1)
    ev_cvp = None
    if F>1:
        ev_cvp = np.sum(node_features[:, -w:, 1], axis=1)
    # prison sums
    pr_sum = np.sum(node_prison[:, -w:], axis=1)

    # compute Pearson across nodes
    def safe_corr(a,b):
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) < 10:
            return None
        a2 = a[mask]; b2 = b[mask]
        if np.nanstd(a2)==0 or np.nanstd(b2)==0:
            return None
        return float(np.corrcoef(a2,b2)[0,1])

    corr_cvli = safe_corr(pr_sum, ev_cvli)
    corr_cvp = safe_corr(pr_sum, ev_cvp) if ev_cvp is not None else None
    res_w['corr_nodes_cvli'] = corr_cvli
    res_w['corr_nodes_cvp'] = corr_cvp
    res_w['nodes_with_prison_events'] = int(np.sum(pr_sum>0))

    # temporal train/test split check: first 70% vs last 30%
    split = int(0.7 * T)
    # sums per node over train/test
    pr_train = np.sum(node_prison[:, :split], axis=1)
    pr_test = np.sum(node_prison[:, split:], axis=1)
    ev_cvli_train = np.sum(node_features[:, :split, 0], axis=1)
    ev_cvli_test = np.sum(node_features[:, split:, 0], axis=1)
    ev_cvp_train = None
    ev_cvp_test = None
    if F>1:
        ev_cvp_train = np.sum(node_features[:, :split, 1], axis=1)
        ev_cvp_test = np.sum(node_features[:, split:, 1], axis=1)

    res_w['corr_train_cvli'] = safe_corr(pr_train, ev_cvli_train)
    res_w['corr_test_cvli'] = safe_corr(pr_test, ev_cvli_test)
    if ev_cvp_train is not None:
        res_w['corr_train_cvp'] = safe_corr(pr_train, ev_cvp_train)
        res_w['corr_test_cvp'] = safe_corr(pr_test, ev_cvp_test)

    results[w] = res_w

out_payload = {
    'generated_at': datetime.utcnow().isoformat(),
    'windows': WINDOWS,
    'results': results,
    'summary': {
        'total_raw_records': len(records),
        'mapped_nodes_with_events': num_nodes_with,
        'unique_node_coords': len(unique_coords)
    }
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(out_payload, fh, indent=2, ensure_ascii=False)

print('Wrote', OUT)
for w, r in results.items():
    print('\nWindow', w)
    print('  corr_nodes_cvli:', r.get('corr_nodes_cvli'))
    print('  corr_nodes_cvp :', r.get('corr_nodes_cvp'))
    print('  train/test cvli:', r.get('corr_train_cvli'), '/', r.get('corr_test_cvli'))
    if 'corr_train_cvp' in r:
        print('  train/test cvp :', r.get('corr_train_cvp'), '/', r.get('corr_test_cvp'))

print('\nDone')

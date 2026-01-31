"""
analyze_prison_by_bairro.py

Aggregate prison events by `bairro_id` (from prisoes_with_features.parquet), align to
model dates and compute Pearson correlations between prison counts and model targets
(CVLI and CVP) for several windows. Save CSV summaries and PNG plots in `plots/`.

Usage: python scripts/analyze_prison_by_bairro.py
"""
import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PACK = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
PARQUET = os.path.join(ROOT, 'data', 'processed', 'prisoes_with_features.parquet')
OUT_DIR = os.path.join(ROOT, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(PARQUET):
    raise RuntimeError('prisoes_with_features.parquet not found; please prepare processed prison features')
if not os.path.exists(PROC_PACK):
    # try graph_data chunks
    gd = os.path.join(ROOT, 'data', 'processed', 'graph_data')
    if not os.path.isdir(gd):
        raise RuntimeError('processed_graph_data.pkl not found and graph_data missing')

print('Loading processed graph data (node_features, dates)')
with open(PROC_PACK, 'rb') as fh:
    try:
        pack = pickle.load(fh)
    except Exception:
        # fallback: load chunks
        pack = {}
        nf = os.path.join(ROOT, 'data', 'processed', 'node_feature_tensor_backup_20260123_105747.npy')
        if os.path.exists(nf):
            arr = np.load(nf, allow_pickle=True)
            # arr may be (T,N,F)
            if arr.ndim==3 and arr.shape[0] != len(arr):
                # attempt to detect orientation using graph_data/dates.pkl
                try:
                    d = pickle.load(open(os.path.join(ROOT,'data','processed','graph_data','dates.pkl'),'rb'))
                    if len(d)==arr.shape[0]:
                        arr = np.transpose(arr, (1,0,2))
                except Exception:
                    pass
            pack['node_features'] = arr
        # try dates
        try:
            pack['dates'] = pickle.load(open(os.path.join(ROOT,'data','processed','graph_data','dates.pkl'),'rb'))
        except Exception:
            pass

node_features = pack.get('node_features')
dates = pack.get('dates')
if node_features is None:
    raise RuntimeError('node_features missing in processed pack')
if dates is None:
    # try graph_data/dates.pkl
    try:
        dates = pickle.load(open(os.path.join(ROOT,'data','processed','graph_data','dates.pkl'),'rb'))
        print('Loaded dates from graph_data/dates.pkl')
    except Exception:
        raise RuntimeError('dates missing in processed pack and graph_data/dates.pkl not available')

# ensure shape (N,T,F)
if node_features.ndim==3:
    # if stored as (T,N,F) where T == len(dates), transpose to (N,T,F)
    if node_features.shape[0] == len(dates):
        node_features = np.transpose(node_features, (1,0,2))
N,T,F = node_features.shape
print(f'Loaded node_features shape (N,T,F): {node_features.shape}')

print('Loading prison parquet')
df = pd.read_parquet(PARQUET)
if 'bairro_id' not in df.columns:
    raise RuntimeError('prisoes_with_features.parquet lacks bairro_id column')
df['date'] = pd.to_datetime(df['Data']).dt.date.astype(str)

# build date index mapping
date_index = {str(d)[:10]: i for i,d in enumerate(dates)}

# build matrix (N x T) of prison counts aggregated by bairro_id
prison_matrix = np.zeros((N, len(dates)), dtype=int)
for (bid, d), grp in df.groupby(['bairro_id','date']):
    try:
        bid = int(bid)
    except Exception:
        continue
    if d not in date_index: continue
    if 0 <= bid < N:
        prison_matrix[bid, date_index[d]] = int(grp.shape[0])

print('Computed prison matrix; nonzero bairros:', int(np.sum(np.any(prison_matrix>0, axis=1))))

WINDOWS = [30,60,90,180]
results = {}

for w in WINDOWS:
    if T < w: continue
    res = {}
    # event sums for CVLI (col 0) and CVP (col 1 if exists)
    ev_cvli = np.sum(node_features[:, -w:, 0], axis=1)
    ev_cvp = np.sum(node_features[:, -w:, 1], axis=1) if F>1 else None
    prison_sum = np.sum(prison_matrix[:, -w:], axis=1)

    def safe_corr(a,b):
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) < 10: return None
        a2 = a[mask]; b2 = b[mask]
        if np.nanstd(a2)==0 or np.nanstd(b2)==0: return None
        return float(np.corrcoef(a2,b2)[0,1])

    res['corr_cvli'] = safe_corr(prison_sum, ev_cvli)
    res['corr_cvp'] = safe_corr(prison_sum, ev_cvp) if ev_cvp is not None else None

    # train/test temporal split (70/30)
    split = int(0.7 * T)
    pr_train = np.sum(prison_matrix[:, :split], axis=1)
    pr_test = np.sum(prison_matrix[:, split:], axis=1)
    ev_cvli_train = np.sum(node_features[:, :split, 0], axis=1)
    ev_cvli_test = np.sum(node_features[:, split:, 0], axis=1)
    res['corr_train_cvli'] = safe_corr(pr_train, ev_cvli_train)
    res['corr_test_cvli'] = safe_corr(pr_test, ev_cvli_test)
    if ev_cvp is not None:
        ev_cvp_train = np.sum(node_features[:, :split, 1], axis=1)
        ev_cvp_test = np.sum(node_features[:, split:, 1], axis=1)
        res['corr_train_cvp'] = safe_corr(pr_train, ev_cvp_train)
        res['corr_test_cvp'] = safe_corr(pr_test, ev_cvp_test)

    results[w] = res

    # save CSV of per-bairro summary
    out_csv = os.path.join(OUT_DIR, f'prison_bairro_summary_{w}d.csv')
    summary = []
    for i in range(N):
        summary.append({'bairro_id':i, 'prison_sum':int(prison_sum[i]), 'cvli_sum':int(ev_cvli[i]), 'cvp_sum': int(ev_cvp[i]) if ev_cvp is not None else None})
    pd.DataFrame(summary).to_csv(out_csv, index=False)

    # plot scatter prison vs cvli
    try:
        plt.figure(figsize=(6,4))
        plt.scatter(prison_sum, ev_cvli, alpha=0.6)
        plt.xlabel('Prison events (sum)')
        plt.ylabel('CVLI events (sum)')
        plt.title(f'Prison vs CVLI (window {w}d) — corr={res["corr_cvli"]}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'prison_vs_cvli_{w}d.png'))
        plt.close()
    except Exception as e:
        print('Plot failed for window', w, e)

OUT = os.path.join('scripts','prison_by_bairro_results.json')
with open(OUT,'w',encoding='utf-8') as fh:
    json.dump({'generated_at':datetime.utcnow().isoformat(), 'windows':WINDOWS, 'results':results}, fh, indent=2, ensure_ascii=False)

print('Wrote', OUT)
print('Results:')
for w,r in results.items():
    print(f'  {w}d: corr_cvli={r.get("corr_cvli")}, corr_cvp={r.get("corr_cvp")}, train/test cvli={r.get("corr_train_cvli")}/{r.get("corr_test_cvli")}')

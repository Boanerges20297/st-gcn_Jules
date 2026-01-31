"""
Compare prison events with model predictions (CVLI & CVP).
Produces correlations for windows [30,60,90,180] and saves JSON + plots.
"""
import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PACK = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
PARQUET = os.path.join(ROOT, 'data', 'processed', 'prisoes_with_features.parquet')
OUT_DIR = os.path.join(ROOT, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# checkpoints
CK_CVLI = os.path.join(ROOT, 'models', 'stgcn_cvli.pth')
CK_CVP = os.path.join(ROOT, 'models', 'stgcn_cvp.pth')

if not os.path.exists(PARQUET):
    raise RuntimeError('prisoes_with_features.parquet not found; please prepare processed prison features')
if not os.path.exists(PROC_PACK):
    raise RuntimeError('processed_graph_data.pkl not found; please run preprocessing')

print('Loading processed graph data')
with open(PROC_PACK,'rb') as fh:
    pack = pickle.load(fh)

node_features = pack.get('node_features')
dates = pack.get('dates')
if node_features is None:
    raise RuntimeError('node_features missing in processed pack')
if dates is None:
    # try graph_data/dates.pkl
    gd = os.path.join(ROOT,'data','processed','graph_data','dates.pkl')
    if os.path.exists(gd):
        dates = pickle.load(open(gd,'rb'))
        print('Loaded dates from graph_data/dates.pkl')
    else:
        raise RuntimeError('dates missing in processed pack and graph_data/dates.pkl not available')
if node_features.ndim==3 and node_features.shape[0]==len(dates):
    node_features = np.transpose(node_features,(1,0,2))
N,T,F = node_features.shape
print('node_features shape', node_features.shape)

# load prison parquet and build prison matrix (N x T)
df = pd.read_parquet(PARQUET)
df['date'] = pd.to_datetime(df['Data']).dt.date.astype(str)
# date map
date_index = {str(d)[:10]: i for i,d in enumerate(dates)}
prison_matrix = np.zeros((N,len(dates)), dtype=int)
for (bid,d),grp in df.groupby(['bairro_id','date']):
    try:
        bid = int(bid)
    except Exception:
        continue
    if d not in date_index: continue
    if 0 <= bid < N:
        prison_matrix[bid, date_index[d]] = int(grp.shape[0])
print('prison matrix built; nonzero bairros:', int(np.sum(np.any(prison_matrix>0,axis=1))))

# helper: safe corr
def safe_corr(a,b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if np.sum(mask) < 10: return None
    a2,b2 = a[mask], b[mask]
    if np.nanstd(a2)==0 or np.nanstd(b2)==0: return None
    return float(np.corrcoef(a2,b2)[0,1])

# function to roll predictions
import sys
sys.path.insert(0, ROOT)
from src.model import STGCN
import torch.nn as nn

def generate_predictions(checkpoint_path, input_series):
    # input_series: (T,N) univariate historical counts
    sd = torch.load(checkpoint_path,map_location='cpu')
    # detect conv_final kernel size time_steps
    # keys like 'conv_final.weight'
    time_steps = None
    for k in sd.keys():
        if k.endswith('conv_final.weight'):
            time_steps = sd[k].shape[-1]
            break
    if time_steps is None:
        raise RuntimeError('could not detect time_steps from checkpoint')
    print('Checkpoint', os.path.basename(checkpoint_path), 'time_steps', time_steps)
    # instantiate model - use input_series node count
    n_model = int(input_series.shape[1])
    model = STGCN(num_nodes=n_model, in_channels=1, time_steps=time_steps, num_classes=1, num_graphs=2)
    # strip module prefixes if any
    new_sd = {}
    for k,v in sd.items():
        nk = k
        if k.startswith('module.'):
            nk = k[len('module.'):]
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    preds = np.full((len(input_series)-time_steps, n_model), np.nan, dtype=float)
    with torch.no_grad():
        for idx in range(time_steps, len(input_series)):
            window = input_series[idx-time_steps:idx]  # shape (time_steps, N)
            # build tensor (1, C=1, N, time_steps)
            x = torch.tensor(window.T[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
            # build dummy adj_list from pack if available
            # build adj_list sized to n_model
            adj_list = []
            # prefer adj_geo/adj_faction from pack when sizes match
            if 'adj_geo' in pack and pack['adj_geo'].shape[0]==n_model:
                adj_list = [torch.tensor(pack['adj_geo'], dtype=torch.float32)]
                if 'adj_faction' in pack and pack['adj_faction'].shape[0]==n_model:
                    adj_list = [torch.tensor(pack['adj_geo'], dtype=torch.float32), torch.tensor(pack['adj_faction'], dtype=torch.float32)]
            else:
                # try adjacency backup
                adj_backup = os.path.join(ROOT,'data','processed','adjacency_matrix_backup_20260123_105747.npy')
                if os.path.exists(adj_backup):
                    a = np.load(adj_backup)
                    if a.shape[0]==n_model:
                        adj_list = [torch.tensor(a, dtype=torch.float32)]
            if not adj_list:
                adj_list = [torch.eye(n_model)]
            # forward: model expects list of adj matrices
            out = model(x, adj_list)
            # out shape (1, N, 1) or (1,N,classes)
            out_np = out.squeeze(0).squeeze(-1).cpu().numpy()
            preds[idx-time_steps,:] = out_np
    return preds, time_steps

# prepare univariate series for CVLI and CVP candidates
cvli_series = np.load(os.path.join(ROOT,'data','processed','tensor_cvli_univariado.npy'))
# ensure shape (T,N)
if cvli_series.shape[0]==T:
    cvli_series = cvli_series
elif cvli_series.shape[1]==T:
    cvli_series = cvli_series.T
else:
    # try transpose fallback
    cvli_series = cvli_series

# for CVP, attempt to use tensor_multivariado channel 1 if present
mult = np.load(os.path.join(ROOT,'data','processed','tensor_multivariado.npy'))
# mult shape (1472,121,3) per inspection; ensure (T,N,C)
if mult.shape[0]==T:
    mult_arr = mult
else:
    # fallback assume (T,N,C)
    mult_arr = mult
# choose channel 1 as cvp candidate
cvp_series = mult_arr[:,:,1]

results = {'generated_at':datetime.utcnow().isoformat(), 'windows':[30,60,90,180], 'models':{}}

# generate predictions for CVLI
pred_cvli, ts_cvli = generate_predictions(CK_CVLI, cvli_series)
# pred_cvli corresponds to dates[ts_cvli:]
print('pred_cvli shape', pred_cvli.shape)
# generate predictions for CVP
pred_cvp, ts_cvp = generate_predictions(CK_CVP, cvp_series)
print('pred_cvp shape', pred_cvp.shape)

# For each window compute aggregated predicted sums aligned to last w days where predictions exist
for w in results['windows']:
    r = {}
    # ensure we have predictions for last w days: preds align to dates[ts:]
    # take last w entries from prediction arrays
    if pred_cvli.shape[0] >= w:
        pred_cvli_sum = np.nansum(pred_cvli[-w:,:], axis=0)
        prison_sum = np.sum(prison_matrix[:pred_cvli_sum.shape[0],-w:], axis=1)
        r['pred_cvli_corr'] = safe_corr(prison_sum, pred_cvli_sum)
    else:
        r['pred_cvli_corr'] = None
    if pred_cvp.shape[0] >= w:
        pred_cvp_sum = np.nansum(pred_cvp[-w:,:], axis=0)
        prison_sum = np.sum(prison_matrix[:pred_cvp_sum.shape[0],-w:], axis=1)
        r['pred_cvp_corr'] = safe_corr(prison_sum, pred_cvp_sum)
    else:
        r['pred_cvp_corr'] = None
    results['models'][w] = r
    # save per-window scatter plots comparing prison_sum and predicted sums if available
    try:
        if r['pred_cvli_corr'] is not None:
            plt.figure(figsize=(6,4))
            plt.scatter(prison_sum, pred_cvli_sum, alpha=0.6)
            plt.xlabel('Prison events (sum)')
            plt.ylabel('Predicted CVLI (sum)')
            plt.title(f'Prison vs Predicted CVLI (window {w}d) — corr={r["pred_cvli_corr"]}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'prison_vs_pred_cvli_{w}d.png'))
            plt.close()
        if r['pred_cvp_corr'] is not None:
            plt.figure(figsize=(6,4))
            plt.scatter(prison_sum, pred_cvp_sum, alpha=0.6)
            plt.xlabel('Prison events (sum)')
            plt.ylabel('Predicted CVP (sum)')
            plt.title(f'Prison vs Predicted CVP (window {w}d) — corr={r["pred_cvp_corr"]}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'prison_vs_pred_cvp_{w}d.png'))
            plt.close()
    except Exception as e:
        print('plot failed', e)

OUT = os.path.join('scripts','prison_vs_predictions_results.json')
with open(OUT,'w',encoding='utf-8') as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)
print('Wrote', OUT)
print('Results:')
for w,r in results['models'].items():
    print(f'  {w}d: pred_cvli_corr={r.get("pred_cvli_corr")}, pred_cvp_corr={r.get("pred_cvp_corr")}')

import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / 'data' / 'processed' / 'processed_graph_data.pkl'
STGCN_PATH = ROOT / 'models' / 'stgcn_model_v2.pth'
RANKING_DIR = ROOT / 'models' / 'ranking_by_day'
OUT = Path('outputs') / 'diagnostics_ranking_integration.json'
OUT.parent.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features_v3_local(ts_window):
    num_nodes = ts_window.shape[0]
    features = np.zeros((num_nodes, 15))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(num_nodes):
            ts = ts_window[i, :]
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.max(ts)
            features[i, 3] = np.min(ts)
            features[i, 4] = np.sum(ts > 0) / len(ts) if len(ts) > 0 else 0
            features[i, 5] = np.sum(ts) / len(ts) if len(ts) > 0 else 0
            if len(ts) > 5:
                features[i, 6] = np.mean(ts[-5:]) - np.mean(ts[:5])
            if len(ts) > 1:
                features[i, 7] = np.mean(np.abs(np.diff(ts)))
            features[i, 8] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 9] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 10] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            if len(ts) > 1:
                mean_val = np.mean(ts)
                if mean_val > 1e-6:
                    features[i, 11] = np.std(ts) / mean_val
            features[i, 12] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0:
                features[i, 13] = (max_val - np.min(ts)) / max_val
    return np.nan_to_num(features)

def normalize(v):
    v = v.astype(float)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

def p_at_k(pred, true, k=5):
    if true.max() == 0:
        return None
    k_actual = min(k, int((true>0).sum()), len(true))
    if k_actual<=0: return None
    pred_top = np.argsort(-pred)[:k_actual]
    true_top = np.argsort(-true)[:k_actual]
    return len(set(pred_top)&set(true_top))/k_actual

def main(num_windows=50):
    with open(DATA_FILE, 'rb') as f: data = pickle.load(f)
    node_features = data['node_features']
    dates = data.get('dates')
    adj_geo = data['adj_geo']
    adj_conflict = data['adj_conflict']

    # load models
    from src.model import STGCN
    stgcn = STGCN(num_nodes=node_features.shape[0], in_channels=node_features.shape[2], time_steps=30, num_graphs=2).to(DEVICE)
    stgcn.load_state_dict(torch.load(str(STGCN_PATH), map_location=DEVICE))
    stgcn.eval()

    # load per-day ranking models and scalers
    rank_models = {}
    for d in range(7):
        sel = RANKING_DIR / f"ranking_model_day{d}_selected.pth"
        alt = RANKING_DIR / f"ranking_model_day{d}.pth"
        path = sel if sel.exists() else (alt if alt.exists() else None)
        scaler_path = RANKING_DIR / f"ranking_scaler_day{d}.pkl"
        scaler = None
        if scaler_path.exists():
            try:
                with open(scaler_path,'rb') as f: scaler = pickle.load(f)
            except Exception:
                scaler = None
        if path is None:
            rank_models[d] = (None, scaler)
            continue
        # local model def
        class M(torch.nn.Module):
            def __init__(self, input_dim=15):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim,128), torch.nn.ReLU(), torch.nn.BatchNorm1d(128), torch.nn.Dropout(0.2),
                    torch.nn.Linear(128,64), torch.nn.ReLU(), torch.nn.Linear(64,1)
                )
            def forward(self,x): return self.net(x)
        m = M().to(DEVICE)
        try:
            m.load_state_dict(torch.load(str(path), map_location=DEVICE))
            m.eval()
        except Exception:
            m = None
        rank_models[d] = (m, scaler)

    # iterate recent windows (validation tail)
    HISTORY_WINDOW=30; HORIZON=7
    total_days = node_features.shape[1]
    valid_range = total_days - HISTORY_WINDOW - HORIZON + 1
    start = max(0, valid_range - num_windows)
    windows = list(range(start, valid_range))

    diagnostics = {'per_window': [], 'per_day_summary': {str(d): {'corrs': [], 'p5_stgcn': [], 'p5_rank': [], 'p5_comb': []} for d in range(7)}}

    # precompute normalized adjs
    def norm_adj(a):
        at = torch.FloatTensor(a)
        rowsum = at.sum(1); d_inv = torch.pow(rowsum, -0.5); d_inv[torch.isinf(d_inv)]=0.
        D = torch.diag(d_inv)
        return torch.mm(torch.mm(D, at), D).to(DEVICE)
    adj_list = [norm_adj(adj_geo), norm_adj(adj_conflict)]

    for s in windows:
        window = node_features[:, s:s+HISTORY_WINDOW, :]
        future = node_features[:, s+HISTORY_WINDOW:s+HISTORY_WINDOW+HORIZON, 0]
        y_true = np.sum(future, axis=1)
        date_target = dates[s+HISTORY_WINDOW] if dates is not None else None
        dow = date_target.weekday() if date_target is not None else 0

        # STGCN
        inp = torch.FloatTensor(np.transpose(window, (2,0,1))[None,...]).to(DEVICE)
        with torch.no_grad():
            out = stgcn(inp, adj_list).squeeze(0).cpu().numpy()[:,0]
        s_norm = normalize(out)

        # ranking
        node_ts = window[:, :, 0]
        feats = extract_features_v3_local(node_ts)
        m, scaler = rank_models.get(dow, (None, None))
        r_scores = np.zeros_like(out)
        if m is not None:
            feats_in = feats
            if scaler is not None:
                try: feats_in = scaler.transform(feats)
                except Exception: feats_in = feats
            with torch.no_grad():
                r_scores = m(torch.FloatTensor(feats_in).to(DEVICE)).cpu().numpy().ravel()
        r_norm = normalize(r_scores)

        # combined (same as before: 0.6/0.4 used in validate_full_pipeline)
        combined = 0.6 * s_norm + 0.4 * r_norm

        # metrics
        p5_s = p_at_k(out, y_true, k=5)
        p5_r = p_at_k(r_scores, y_true, k=5)
        p5_c = p_at_k(combined, y_true, k=5)
        # correlations
        try:
            pear = float(np.corrcoef(s_norm, r_norm)[0,1])
        except Exception:
            pear = None
        try:
            from scipy.stats import spearmanr
            spear = float(spearmanr(s_norm, r_norm).correlation)
        except Exception:
            spear = None

        diagnostics['per_window'].append({'window_index': s, 'date': str(date_target.date()) if date_target is not None else None, 'dow': dow, 'p5_stgcn': p5_s, 'p5_rank': p5_r, 'p5_comb': p5_c, 'pearson': pear, 'spearman': spear, 'r_stats': {'min': float(r_scores.min()), 'max': float(r_scores.max()), 'mean': float(r_scores.mean()), 'std': float(r_scores.std())}})
        dsum = diagnostics['per_day_summary'][str(dow)]
        if pear is not None: dsum['corrs'].append(pear)
        if p5_s is not None: dsum['p5_stgcn'].append(p5_s)
        if p5_r is not None: dsum['p5_rank'].append(p5_r)
        if p5_c is not None: dsum['p5_comb'].append(p5_c)

    # aggregate per day
    for d, v in diagnostics['per_day_summary'].items():
        v['mean_corr'] = float(np.mean(v['corrs'])) if len(v['corrs'])>0 else None
        v['mean_p5_stgcn'] = float(np.mean(v['p5_stgcn'])) if len(v['p5_stgcn'])>0 else None
        v['mean_p5_rank'] = float(np.mean(v['p5_rank'])) if len(v['p5_rank'])>0 else None
        v['mean_p5_comb'] = float(np.mean(v['p5_comb'])) if len(v['p5_comb'])>0 else None
        # drop raw lists to keep file small
        del v['corrs']; del v['p5_stgcn']; del v['p5_rank']; del v['p5_comb']

    with open(OUT, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print('Saved diagnostics to', OUT)

if __name__ == '__main__':
    main(num_windows=50)

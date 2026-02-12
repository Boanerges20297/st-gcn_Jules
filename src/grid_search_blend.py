import os
import sys
import json
import pickle
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import STGCN

DATA_PATH = ROOT / 'data' / 'processed' / 'processed_graph_data.pkl'
MODEL_PATH = ROOT / 'models' / 'stgcn_model_v2.pth'
SCALER_PATH = ROOT / 'models' / 'ranking_by_day' / 'ranking_scaler.pkl'
RANKING_PATH = ROOT / 'models' / 'ranking_by_day' / 'ranking_model_day2_selected.pth'
OUT_PATH = Path('outputs') / 'blend_grid_search.json'
OUT_PATH.parent.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features_v3(ts_window):
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
            if len(ts) > 5: features[i, 6] = np.mean(ts[-5:]) - np.mean(ts[:5])
            if len(ts) > 1: features[i, 7] = np.mean(np.abs(np.diff(ts)))
            features[i, 8] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 9] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 10] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            if len(ts) > 1:
                mean_val = np.mean(ts)
                if mean_val > 1e-6: features[i, 11] = np.std(ts) / mean_val
            features[i, 12] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0: features[i, 13] = (max_val - np.min(ts)) / max_val
    return np.nan_to_num(features)

def normalize(v):
    v = v.astype(float)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

class RankingModelV3(torch.nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.BatchNorm1d(hidden_dim), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def evaluate(blend_stgcn, blend_rank, test_weeks=8):
    with open(DATA_PATH, 'rb') as f: data = pickle.load(f)
    node_features = data['node_features']
    dates = data['dates']
    nodes_gdf = data.get('nodes_gdf')

    # load ST-GCN
    num_nodes = node_features.shape[0]
    stgcn = STGCN(num_nodes=num_nodes, in_channels=node_features.shape[2], time_steps=30, num_graphs=2).to(DEVICE)
    stgcn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    stgcn.eval()

    # load ranking model and scaler (if exists)
    scaler = None
    if SCALER_PATH.exists():
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    rank_model = None
    if RANKING_PATH.exists():
        rank_model = RankingModelV3().to(DEVICE)
        rank_model.load_state_dict(torch.load(RANKING_PATH, map_location=DEVICE))
        rank_model.eval()

    def get_p_at_k(y_true, y_pred, k=5):
        idx_true = np.argsort(y_true)[-k:]
        idx_pred = np.argsort(y_pred)[-k:]
        return len(set(idx_true) & set(idx_pred)) / k

    metrics = []
    total_days = len(dates)
    window = 30; horizon = 7
    for w in range(test_weeks):
        end_idx = total_days - horizon - (w * 7)
        start_idx = end_idx - window
        if start_idx < 0: break
        X_slice = node_features[:, start_idx:end_idx, :]
        X_tensor = torch.FloatTensor(X_slice).permute(2,0,1).unsqueeze(0).to(DEVICE)
        future_slice = node_features[:, end_idx:end_idx+horizon, 0]
        y_true_full = np.sum(future_slice, axis=1)
        if np.sum(y_true_full) == 0: continue

        with torch.no_grad():
            pred = stgcn(X_tensor, [torch.FloatTensor(data['adj_geo']).to(DEVICE), torch.FloatTensor(data['adj_conflict']).to(DEVICE)])
            stgcn_score = pred.squeeze(0).cpu().numpy()[:,0]

        final_score = stgcn_score
        if rank_model is not None and scaler is not None:
            feats = extract_features_v3(X_slice[:, :, 0])
            feats_scaled = scaler.transform(feats)
            with torch.no_grad():
                rank_score = rank_model(torch.FloatTensor(feats_scaled).to(DEVICE)).cpu().numpy()[:,0]
            stgcn_norm = normalize(stgcn_score)
            rank_norm = normalize(rank_score)
            final_score = stgcn_norm * blend_stgcn + rank_norm * blend_rank

        # filter region if available
        target_indices = list(range(num_nodes))
        if nodes_gdf is not None:
            # keep all if nodes_gdf not matching
            pass

        y_true = y_true_full[target_indices]
        y_pred = final_score[target_indices]
        p5 = get_p_at_k(y_true, y_pred, k=5)
        metrics.append(p5)

    return float(np.mean(metrics)) if len(metrics)>0 else 0.0

def main():
    weights = np.arange(0.5, 1.01, 0.05)
    results = {}
    for w in weights:
        w_stgcn = float(w)
        w_rank = float(max(0.0, 1.0 - w_stgcn))
        p5 = evaluate(w_stgcn, w_rank)
        print(f"w_stgcn={w_stgcn:.2f} w_rank={w_rank:.2f} -> P@5={p5:.4f}")
        results[f"{w_stgcn:.2f}"] = {'w_rank': w_rank, 'p5': p5}

    best_w = max(results.items(), key=lambda x: x[1]['p5'])
    out = {'results': results, 'best': best_w}
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved grid search to', OUT_PATH)

if __name__ == '__main__':
    main()

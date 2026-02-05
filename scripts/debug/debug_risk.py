import os
import sys
import pickle
import torch
import numpy as np

# garantir que o pacote src esteja no path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import STGCN

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_model.pth')

def load_data():
    with open(DATA_FILE, 'rb') as f:
        d = pickle.load(f)
    return d

def load_model(num_nodes, in_channels, time_steps, device='cpu'):
    m = STGCN(num_nodes=num_nodes, in_channels=in_channels, time_steps=time_steps)
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m

def run_window(window):
    d = load_data()
    node_features = d['node_features']  # (N, T, F)
    adj = d['adj_matrix']
    N, T, F = node_features.shape
    dates = d.get('dates')
    if dates is not None:
        dates = list(dates)
        # print last 30 days totals
        daily_cvli = np.sum(node_features[:, -30:, 0], axis=0)
        daily_cvp = np.sum(node_features[:, -30:, 1], axis=0)
        print('Last 30 days CVLI totals (most recent last):')
        for i in range(len(daily_cvli)):
            idx = -30 + i
            print(f"{dates[idx].date()}: {int(daily_cvli[i])} cvli, {int(daily_cvp[i])} cvp")
    w = min(window, T)
    recent = node_features[:, -w:, :]
    # per-node sums in the window
    per_node_cvli = np.sum(node_features[:, -w:, 0], axis=1)
    print('nodes with any CVLI in window:', int(np.sum(per_node_cvli>0)), '/', N)
    device = torch.device('cpu')
    try:
        model = load_model(num_nodes=N, in_channels=F, time_steps=w, device=device)
    except Exception as e:
        print(f'Could not load model with time_steps={w}: {e}')
        return

    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt)

    inp = torch.FloatTensor(recent).permute(2, 0, 1).unsqueeze(0).to(device)
    print('input tensor stats: sum=', float(inp.sum()), 'mean=', float(inp.mean()), 'nonzero=', int(torch.count_nonzero(inp)))
    # inspect some layer weights
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv_final' not in name:
            print(f'{name} sum:', float(param.sum()))

    with torch.no_grad():
        pred = model(inp, norm_adj)

        # inspect conv_final weights
    conv_w = None
    for k,v in model.state_dict().items():
        if 'conv_final.weight' in k:
            conv_w = v
            break
    if conv_w is not None:
        print('conv_final weight sum:', float(conv_w.sum()), 'mean:', float(conv_w.mean()))

    pred = pred.squeeze(0).cpu().numpy()  # (N, 2)
    print('raw pred min/max sample[0:10]:', float(np.min(pred)), float(np.max(pred)), pred[:10])
    # predictions may be negative; keep raw for inspection
    pred_nonneg = np.maximum(pred, 0)
    cvli = pred[:, 0]
    cvli_adj = cvli * 1.2
    # new shift-based normalization (same approach used in app.py)
    risk_score = cvli_adj
    min_risk = np.min(risk_score)
    shifted = risk_score - min_risk
    max_shift = np.max(shifted) if np.max(shifted) > 0 else 1
    normalized_shift = (shifted / max_shift) * 100
    cutoff = float(np.percentile(normalized_shift, 95))
    print('After shift-normalize: max=', float(np.max(normalized_shift)), 'cutoff95=', cutoff,
          'priority_count=', int(np.sum(normalized_shift >= cutoff)))
    max_risk = np.max(cvli_adj) if np.max(cvli_adj) > 0 else 1
    normalized = (cvli_adj / max_risk) * 100

    def stats(arr):
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p90': float(np.percentile(arr,90)),
            'p99': float(np.percentile(arr,99)),
            'count>0.01': int(np.sum(arr>0.01)),
            'count>0.001': int(np.sum(arr>0.001)),
            'count>0': int(np.sum(arr>0))
        }

    print(f"Window={w}: N={N}, time_steps_total={T}")
    print('cvli raw stats:', stats(cvli))
    print('cvli_adj stats:', stats(cvli_adj))
    print('normalized stats:', stats(normalized))
    # how many nodes exceed top percentiles
    print('nodes above 95 percentile (normalized):', int(np.sum(normalized >= np.percentile(normalized,95))))

if __name__ == '__main__':
    for w in (7, 14, 30, 90, 180, 365):
        run_window(w)
        print('---')

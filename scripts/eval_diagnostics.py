#!/usr/bin/env python3
"""Diagnostics for CVLI evaluation: prints stats and sample comparisons.

Creates flattened metrics across evaluation steps and shows the first 50
node/time comparisons (y_true, y_pred, abs_error). Useful to diagnose R2
anomalies and scale mismatches.
"""
import os
import pickle
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import numpy as np
import torch

from src.model import STGCN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvli.pth')

WINDOW = 180
EVAL_STEPS = 50

def parse_args():
    p = argparse.ArgumentParser(description='CVLI diagnostics')
    p.add_argument('--auto-desscale', action='store_true', help='Fit linear mapping from preds->true and report desscaled metrics')
    return p.parse_args()

def load_arrays():
    nf_path = os.path.join(DATA_DIR, 'node_features.npy')
    adj_path = os.path.join(DATA_DIR, 'adj_matrix.npy')
    if not os.path.exists(nf_path):
        raise FileNotFoundError(nf_path)
    node_features = np.load(nf_path, allow_pickle=False)
    if os.path.exists(adj_path):
        adj = np.load(adj_path, allow_pickle=True)
    else:
        adj = None
    dates = None
    dpkl = os.path.join(DATA_DIR, 'dates.pkl')
    if os.path.exists(dpkl):
        try:
            with open(dpkl, 'rb') as fh:
                dates = pickle.load(fh)
        except Exception:
            dates = None
    return node_features, adj, dates

def normalize_adj(a):
    if a is None:
        return None
    A = torch.FloatTensor(a)
    rowsum = A.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(D, A), D)

def main():
    args = parse_args()
    node_features, adj, dates = load_arrays()
    N, T, C = node_features.shape
    print(f"Loaded node_features shape: {node_features.shape}")
    if adj is None:
        adj = np.eye(N)

    device = torch.device('cpu')
    norm_adj = normalize_adj(adj).to(device)

    model = STGCN(num_nodes=N, in_channels=1, time_steps=WINDOW, num_classes=1, num_graphs=1)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and 'state_dict' in state:
                model.load_state_dict(state['state_dict'], strict=False)
        model.to(device)
        model.eval()
    else:
        raise FileNotFoundError(MODEL_PATH)

    start_idx = WINDOW
    eval_points = list(range(max(start_idx, T - EVAL_STEPS), T))
    print(f"Evaluating {len(eval_points)} steps (indices {eval_points[:3]} ... {eval_points[-1]})")

    flat_y_true = []
    flat_y_pred = []
    coords = []  # (node_idx, time_idx)
    per_step_stats = []

    for t in eval_points:
        X = node_features[:, t-WINDOW:t, 0:1]
        y_true = node_features[:, t, 0]
        inp = torch.FloatTensor(X).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp, [norm_adj])
        pred_np = pred.squeeze(0).cpu().numpy().reshape(-1)

        mae = mean_absolute_error(y_true, pred_np)
        rmse = math.sqrt(mean_squared_error(y_true, pred_np))
        try:
            r2 = r2_score(y_true, pred_np)
        except Exception:
            r2 = float('nan')
        per_step_stats.append({'t': t, 'mae': mae, 'rmse': rmse, 'r2': r2})

        for i in range(len(y_true)):
            flat_y_true.append(float(y_true[i]))
            flat_y_pred.append(float(pred_np[i]))
            coords.append((i, t))

    y_true_arr = np.array(flat_y_true)
    y_pred_arr = np.array(flat_y_pred)
    errors = np.abs(y_true_arr - y_pred_arr)

    print('\nGlobal statistics for y_true:')
    print(f'  count: {y_true_arr.size}')
    print(f'  mean: {y_true_arr.mean():.6f}')
    print(f'  var: {y_true_arr.var():.6f}')
    print(f'  std: {y_true_arr.std():.6f}')
    print(f'  min/max: {y_true_arr.min():.6f} / {y_true_arr.max():.6f}')

    print('\nGlobal prediction errors:')
    print(f'  MAE: {errors.mean():.6f}')
    print(f'  RMSE: {np.sqrt(((y_true_arr - y_pred_arr)**2).mean()):.6f}')
    try:
        global_r2 = r2_score(y_true_arr, y_pred_arr)
        print(f'  R2: {global_r2:.6f}')
    except Exception:
        print('  R2: nan')

    baseline = np.mean(y_true_arr)
    baseline_preds = np.full_like(y_true_arr, baseline)
    baseline_mae = mean_absolute_error(y_true_arr, baseline_preds)
    print(f'  Baseline (predict mean={baseline:.6f}) MAE: {baseline_mae:.6f}')

    # optional auto-desscale: fit linear map pred' = a*pred + b
    if args.auto_desscale:
        try:
            out_dir = os.path.join(BASE_DIR, 'reports')
            os.makedirs(out_dir, exist_ok=True)
            P = y_pred_arr
            Y = y_true_arr
            a, b = np.polyfit(P, Y, 1)
            fh_map = os.path.join(out_dir, 'desscale_mapping.txt')
            print(f'Auto-desscale mapping: y = {a:.6f} * pred + {b:.6f}')
            with open(fh_map, 'w', encoding='utf-8') as fhm:
                fhm.write(f'a={a}\nb={b}\n')
            # recompute metrics
            pred_d = (a * y_pred_arr) + b
            mae_d = mean_absolute_error(y_true_arr, pred_d)
            rmse_d = math.sqrt(mean_squared_error(y_true_arr, pred_d))
            try:
                r2_d = r2_score(y_true_arr, pred_d)
            except Exception:
                r2_d = float('nan')
            print('\nDESSCALED GLOBAL metrics:')
            print(f'  MAE: {mae_d:.6f}')
            print(f'  RMSE: {rmse_d:.6f}')
            print(f'  R2: {r2_d:.6f}')
        except Exception as e:
            print('Auto-desscale failed:', e)

    print('\nPer-step summary (first 5 steps):')
    for s in per_step_stats[:5]:
        print(f"  t={s['t']}: mae={s['mae']:.6f}, rmse={s['rmse']:.6f}, r2={s['r2']:.6f}")

    print('\nFirst 50 comparisons (node_idx, time_idx, y_true, y_pred, abs_error):')
    for k in range(min(50, y_true_arr.size)):
        ni, ti = coords[k]
        print(f"{k+1:02d}: node={ni}, t={ti}, y_true={y_true_arr[k]:.6f}, y_pred={y_pred_arr[k]:.6f}, err={errors[k]:.6f}")

    # Save report
    out_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'eval_diagnostics.txt')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('Global y_true stats:\n')
        fh.write(f'count: {y_true_arr.size}\n')
        fh.write(f'mean: {y_true_arr.mean():.6f}\n')
        fh.write(f'var: {y_true_arr.var():.6f}\n')
        fh.write(f'min/max: {y_true_arr.min():.6f} / {y_true_arr.max():.6f}\n')
        fh.write('\nPrediction errors:\n')
        fh.write(f'MAE: {errors.mean():.6f}\n')
        fh.write(f'RMSE: {np.sqrt(((y_true_arr - y_pred_arr)**2).mean()):.6f}\n')
        try:
            fh.write(f'R2: {global_r2:.6f}\n')
        except Exception:
            fh.write('R2: nan\n')
        fh.write(f'Baseline MAE (predict mean={baseline:.6f}): {baseline_mae:.6f}\n')
        fh.write('\nFirst 50 comparisons:\n')
        for k in range(min(50, y_true_arr.size)):
            ni, ti = coords[k]
            fh.write(f"{k+1:02d}: node={ni}, t={ti}, y_true={y_true_arr[k]:.6f}, y_pred={y_pred_arr[k]:.6f}, err={errors[k]:.6f}\n")

    print(f'\nDetailed report saved to {out_path}')

if __name__ == '__main__':
    main()

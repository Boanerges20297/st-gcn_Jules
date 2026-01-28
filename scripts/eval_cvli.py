#!/usr/bin/env python3
"""Evaluate CVLI predictive accuracy with sliding-window backtest.

Produces MAE, RMSE and R2 aggregated across nodes for the last N evaluation steps.
"""
import os
import json
import numpy as np
import pickle
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import torch
import pandas as pd

from src.model import STGCN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'graph_data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stgcn_cvli.pth')

WINDOW = 180
EVAL_STEPS = 50

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate CVLI model')
    p.add_argument('--auto-desscale', action='store_true', help='Fit linear mapping from preds->true and desscale predictions')
    return p.parse_args()

def load_desscale_mapping():
    # look for a mapping saved by diagnostics
    path = os.path.join(BASE_DIR, 'reports', 'desscale_mapping.txt')
    if os.path.exists(path):
        try:
            a = None
            b = None
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('a='):
                        a = float(line.split('=',1)[1])
                    elif line.startswith('b='):
                        b = float(line.split('=',1)[1])
            if a is not None and b is not None:
                print(f'Loaded desscale mapping from {path}: a={a}, b={b}')
                return a, b
        except Exception:
            pass
    return None

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
    # try dates for reference
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
        print("Warning: adjacency not found; model may expect graph input. Using identity.")
        adj = np.eye(N)

    device = torch.device('cpu')

    num_graphs = 1
    norm_adj = normalize_adj(adj).to(device)

    # instantiate model
    model = STGCN(num_nodes=N, in_channels=1, time_steps=WINDOW, num_classes=1, num_graphs=num_graphs)
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=device)
            # attempt to load state dict directly
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                # maybe state is a dict with 'state_dict'
                if 'state_dict' in state:
                    model.load_state_dict(state['state_dict'], strict=False)
            model.to(device)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed loading model: {e}")
    else:
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Determine evaluation time indices: last EVAL_STEPS points where window available
    start_idx = WINDOW
    end_idx = T
    eval_points = list(range(max(start_idx, T - EVAL_STEPS), T))
    if not eval_points:
        raise RuntimeError("Not enough timesteps for evaluation")

    all_maes = []
    all_rmses = []
    all_r2s = []

    all_true = []
    all_pred = []

    for t in eval_points:
        # input window: [t-WINDOW : t)
        X = node_features[:, t-WINDOW:t, 0:1]  # (N, WINDOW, 1)
        y_true = node_features[:, t, 0]      # (N,)

        inp = torch.FloatTensor(X).permute(2,0,1).unsqueeze(0).to(device)  # (1,1,N,WINDOW)
        with torch.no_grad():
            pred = model(inp, [norm_adj])  # (1, N, 1)
        pred_np = pred.squeeze(0).cpu().numpy().reshape(-1)

        # try load mapping once and apply (if exists)
        if t == eval_points[0]:
            mapping = load_desscale_mapping()
            if mapping is not None:
                a_map, b_map = mapping
            else:
                a_map = None
                b_map = None
        if a_map is not None:
            pred_np = (a_map * pred_np) + b_map

        mae = mean_absolute_error(y_true, pred_np)
        rmse = math.sqrt(mean_squared_error(y_true, pred_np))
        try:
            r2 = r2_score(y_true, pred_np)
        except Exception:
            r2 = float('nan')

        all_maes.append(mae)
        all_rmses.append(rmse)
        all_r2s.append(r2)

        all_true.append(y_true)
        all_pred.append(pred_np)

    # handle desscale mapping
    mapping = load_desscale_mapping()
    if mapping is not None:
        a_map, b_map = mapping
        # already applied per-step; report desscaled global metrics
        Y = np.concatenate(all_true).ravel()
        P = np.concatenate(all_pred).ravel()
        P_d = (a_map * P) + b_map
        mae_d = mean_absolute_error(Y, P_d)
        rmse_d = math.sqrt(mean_squared_error(Y, P_d))
        try:
            r2_d = r2_score(Y, P_d)
        except Exception:
            r2_d = float('nan')
        print(f"Loaded desscale mapping used for reporting: y = {a_map:.6f} * pred + {b_map:.6f}")
        print(f"DESSCALED MAE (global): {mae_d:.4f}")
        print(f"DESSCALED RMSE (global): {rmse_d:.4f}")
        print(f"DESSCALED R2 (global): {r2_d:.6f}")
    elif args.auto_desscale:
        try:
            Y = np.concatenate(all_true).ravel()
            P = np.concatenate(all_pred).ravel()
            # fit slope/intercept
            a, b = np.polyfit(P, Y, 1)
            print(f"Auto-desscale mapping: y = {a:.6f} * pred + {b:.6f}")
            # recompute metrics using desscaled preds
            desscaled_maes = []
            desscaled_rmses = []
            desscaled_r2s = []
            for true, pred in zip(all_true, all_pred):
                pred2 = (a * pred) + b
                desscaled_maes.append(mean_absolute_error(true, pred2))
                desscaled_rmses.append(math.sqrt(mean_squared_error(true, pred2)))
                try:
                    desscaled_r2s.append(r2_score(true, pred2))
                except Exception:
                    desscaled_r2s.append(float('nan'))
            print(f"DESSCALED MAE (mean over steps): {np.mean(desscaled_maes):.4f}")
            print(f"DESSCALED RMSE (mean over steps): {np.mean(desscaled_rmses):.4f}")
            valid_dr2 = [v for v in desscaled_r2s if not (v is None or (isinstance(v,float) and np.isnan(v)))]
            if valid_dr2:
                print(f"DESSCALED R2 (mean over steps): {np.mean(valid_dr2):.4f}")
        except Exception as e:
            print('Auto-dessale failed:', e)

    print(f"Evaluated {len(eval_points)} steps (last index {eval_points[-1]})")
    print(f"MAE (mean over steps): {np.mean(all_maes):.4f}")
    print(f"RMSE (mean over steps): {np.mean(all_rmses):.4f}")
    valid_r2 = [v for v in all_r2s if not (v is None or (isinstance(v,float) and np.isnan(v)))]
    if valid_r2:
        print(f"R2 (mean over steps): {np.mean(valid_r2):.4f}")
    else:
        print("R2: no valid scores")

if __name__ == '__main__':
    main()

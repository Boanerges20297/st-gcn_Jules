import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

# Ensure src is in path
sys.path.append(os.getcwd())

from src.model import STGCN

def normalize_adj(adj):
    # Standard GCN normalization
    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt)
    return norm_adj

def evaluate_top_k(pred_scores, actual_scores, k=30, nodes_gdf=None):
    # Ensure inputs are numpy arrays
    pred_scores = np.array(pred_scores).flatten()
    actual_scores = np.array(actual_scores).flatten()

    # Get indices of top k (descending order)
    top_k_pred = np.argsort(pred_scores)[-k:][::-1]
    top_k_actual = np.argsort(actual_scores)[-k:][::-1]

    # Intersection
    intersection_set = set(top_k_pred) & set(top_k_actual)
    intersection = len(intersection_set)
    precision = intersection / k

    # Risk Capture
    total_events = np.sum(actual_scores)
    captured_events = np.sum(actual_scores[top_k_pred])
    risk_capture = captured_events / total_events if total_events > 0 else 0

    result = {
        'precision': precision,
        'risk_capture': risk_capture,
        'intersection_count': intersection,
        'top_k_pred': top_k_pred,
        'top_k_actual': top_k_actual,
        'total_events': total_events,
        'captured_events': captured_events
    }

    if nodes_gdf is not None and 'name' in nodes_gdf.columns:
        result['pred_names'] = nodes_gdf.iloc[top_k_pred]['name'].values
        result['actual_names'] = nodes_gdf.iloc[top_k_actual]['name'].values
        result['intersection_names'] = nodes_gdf.iloc[list(intersection_set)]['name'].values

    return result

def run_validation():
    print("Loading processed data...")
    data_path = 'data/processed/processed_graph_data.pkl'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    node_features = data['node_features'] # (N, T, C)
    adj_geo = data['adj_matrix'] # (N, N)
    dates = pd.to_datetime(data['dates'])
    nodes_gdf = data.get('nodes_gdf')

    N, T, C = node_features.shape
    print(f"Data loaded: {N} nodes, {T} timesteps.")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")

    # Define Cutoff
    cutoff_date = pd.Timestamp('2026-01-21')

    if cutoff_date not in dates.values:
        print(f"Error: Cutoff date {cutoff_date} not found in data dates.")
        # Find closest
        closest_idx = np.abs(dates - cutoff_date).argmin()
        cutoff_date = dates[closest_idx]
        print(f"Using closest date: {cutoff_date}")

    cutoff_idx = np.where(dates == cutoff_date)[0][0]
    print(f"Cutoff Index: {cutoff_idx} ({dates[cutoff_idx].date()})")

    # Prepare Adjacency
    norm_adj = normalize_adj(adj_geo)
    # Model expects list of 2 (replicated)
    adj_list = [norm_adj, norm_adj]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_list = [a.to(device) for a in adj_list]

    # --- CVLI Validation ---
    print("\n=== CVLI VALIDATION (7 Days Horizon) ===")
    history_window = 90  # Corrected from checkpoint mismatch
    horizon = 7
    feature_idx = 0

    # Check bounds
    if cutoff_idx < history_window:
        print("Not enough history for CVLI.")
    elif cutoff_idx + horizon >= T:
        print(f"Not enough future data for CVLI validation. Need {horizon} days after {cutoff_date.date()}. Data ends {dates.max().date()}.")
        # Try to use whatever is available?
        actual_horizon = T - 1 - cutoff_idx
        if actual_horizon <= 0:
             print("No future data at all.")
        else:
             print(f"Using available horizon: {actual_horizon} days")
             horizon = actual_horizon

    if horizon > 0:
        # Input: [cutoff - history + 1 : cutoff + 1] (Includes the cutoff date)
        input_seq = node_features[:, cutoff_idx - history_window + 1 : cutoff_idx + 1, feature_idx:feature_idx+1]
        # Shape: (N, 120, 1) -> Model needs (B, C, N, T) -> (1, 1, N, 120)
        # Transpose input_seq to (1, N, 120) first?
        # Wait, model expects:
        # x shape: (batch, N, in_features, T) ??
        # Let's check model.py:
        # STGCNLayer: x = self.temporal_conv(x) -> Conv2d(in, out, (1, k))
        # input x to model is passed to layer1.
        # usually (B, C, N, T) for Conv2d where H=N, W=T.

        # train_separate.py: X.append(np.transpose(window, (2, 0, 1))) -> window was (N, T, C)
        # So window (N, T, 1) -> transpose(2, 0, 1) -> (1, N, T).
        # So X shape is (C, N, T).
        # Loader yields (B, C, N, T). Correct.

        input_tensor = np.transpose(input_seq, (2, 0, 1)) # (1, N, 120)
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device) # (1, 1, N, 120)

        # Load Model
        model_path = 'models/stgcn_cvli.pth'
        if os.path.exists(model_path):
            # Model appears to be trained with num_graphs=1 (missing weights.1 in checkpoint)
            model = STGCN(num_nodes=N, in_channels=1, time_steps=history_window, num_classes=1, num_graphs=1).to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    # Pass list of 1 adjacency matrix if num_graphs=1
                    pred = model(input_tensor, [adj_list[0]])
                    # Model forward: x = self.fc(x) -> (B, N, num_classes)

                pred_scores = pred.squeeze().cpu().numpy() # (N,)

                # Ground Truth
                # Sum of events from cutoff (inclusive? No, prediction is for NEXT steps)
                # train_separate.py: target = np.sum(features[:, i+history_window : i+history_window+horizon, :], axis=1)
                # so if i points to start of window, i+history is the first predicted day.
                # So here, if we feed [cutoff-120 : cutoff], we predict [cutoff : cutoff+7].

                # Ground Truth: [cutoff + 1 : cutoff + 1 + horizon] (Starts day after cutoff)
                gt_data = node_features[:, cutoff_idx + 1 : cutoff_idx + 1 + horizon, feature_idx]
                actual_scores = np.sum(gt_data, axis=1)

                # Metrics
                metrics = evaluate_top_k(pred_scores, actual_scores, k=30, nodes_gdf=nodes_gdf)

                print(f"Target Period: {dates[cutoff_idx + 1].date()} to {dates[cutoff_idx + horizon].date()}")
                print(f"Total Actual CVLI Events: {metrics['total_events']}")
                print(f"Precision@30: {metrics['precision']:.2%} ({metrics['intersection_count']}/30)")
                print(f"Risk Capture: {metrics['risk_capture']:.2%}")

                if 'pred_names' in metrics:
                    print("\nTop 5 Predicted Areas (Sample):")
                    for name in metrics['pred_names'][:5]:
                        print(f" - {name}")

                print(f"\nTop 30 Overlap (Common Critical Areas): {metrics['intersection_count']}")

            except Exception as e:
                print(f"Error running CVLI model: {e}")
        else:
            print(f"CVLI Model not found at {model_path}")

    # --- CVP Validation ---
    print("\n=== CVP VALIDATION (1 Day Horizon) ===")
    history_window = 30  # Corrected from checkpoint mismatch
    horizon = 1
    feature_idx = 1

    if cutoff_idx < history_window:
        print("Not enough history for CVP.")
    elif cutoff_idx + horizon > T:
        print("Not enough future data for CVP.")
    else:
        # Input: [cutoff - history + 1 : cutoff + 1]
        input_seq = node_features[:, cutoff_idx - history_window + 1 : cutoff_idx + 1, feature_idx:feature_idx+1]
        input_tensor = np.transpose(input_seq, (2, 0, 1))
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)

        model_path = 'models/stgcn_cvp.pth'
        if os.path.exists(model_path):
            model = STGCN(num_nodes=N, in_channels=1, time_steps=history_window, num_classes=1, num_graphs=1).to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    pred = model(input_tensor, [adj_list[0]])

                pred_scores = pred.squeeze().cpu().numpy()

                # GT: [cutoff + 1 : cutoff + 1 + horizon]
                gt_data = node_features[:, cutoff_idx + 1 : cutoff_idx + 1 + horizon, feature_idx]
                actual_scores = np.sum(gt_data, axis=1) # Just the one day

                metrics = evaluate_top_k(pred_scores, actual_scores, k=30, nodes_gdf=nodes_gdf)

                print(f"Target Date: {dates[cutoff_idx + 1].date()}")
                print(f"Total Actual CVP Events: {metrics['total_events']}")
                print(f"Precision@30: {metrics['precision']:.2%} ({metrics['intersection_count']}/30)")
                print(f"Risk Capture: {metrics['risk_capture']:.2%}")

            except Exception as e:
                print(f"Error running CVP model: {e}")
        else:
            print(f"CVP Model not found at {model_path}")

if __name__ == "__main__":
    run_validation()

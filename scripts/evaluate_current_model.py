import time
import logging
import torch
import torch.nn as nn
import pickle
import numpy as np
import os
import sys

# Add parent directory to path to import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import STGCN

DATA_FILE = 'data/processed/processed_graph_data.pkl'
MODEL_PATH = 'models/stgcn_model_v2.pth'
HISTORY_WINDOW = 30
HORIZON = 7
BATCH_SIZE = 64

# --- Metric & Loss Definitions (Copied from src/train.py for standalone execution) ---

class WeightedFocalMSELoss(nn.Module):
    def __init__(self, weight_zero=1.0, weight_hotspot=20.0, gamma=1.5):
        super(WeightedFocalMSELoss, self).__init__()
        self.weight_zero = weight_zero
        self.weight_hotspot = weight_hotspot
        self.gamma = gamma

    def forward(self, pred, target):
        squared_error = (pred - target) ** 2
        weights = torch.ones_like(target) * self.weight_zero
        weights[target > 0] = self.weight_hotspot
        abs_error = torch.abs(pred - target)
        focal_term = (1 + abs_error) ** self.gamma
        loss = weights * focal_term * squared_error
        return torch.mean(loss)

def precision_at_k(pred, target, k=5):
    """
    Calculates P@K.
    """
    batch_size = pred.shape[0]
    p_k_sum = 0.0
    valid_days = 0

    for i in range(batch_size):
        p = pred[i, :, 0].detach().cpu().numpy()
        t = target[i, :, 0].detach().cpu().numpy()

        if t.max() == 0:
            continue

        valid_days += 1
        num_events = (t > 0).sum()
        k_actual = min(k, num_events, len(t))

        if k_actual <= 0:
            continue

        _, true_top_k_indices = torch.topk(torch.FloatTensor(t), k_actual)
        true_top_k_indices = true_top_k_indices.numpy()

        pred_top_k = torch.topk(torch.FloatTensor(p), k_actual)[1].numpy()

        hits = len(set(true_top_k_indices) & set(pred_top_k))
        p_k_sum += (hits / k_actual)

    return p_k_sum / max(1, valid_days)

def prepare_dataset(node_features):
    num_nodes, num_timesteps, num_features = node_features.shape
    valid_range = num_timesteps - HISTORY_WINDOW - HORIZON + 1
    X_list = []
    Y_list = []

    for s in range(valid_range):
        window = node_features[:, s:s+HISTORY_WINDOW, :]
        target = np.sum(node_features[:, s+HISTORY_WINDOW:s+HISTORY_WINDOW+HORIZON, 0:1], axis=1)
        X_list.append(np.transpose(window, (2, 0, 1)).astype(np.float32))
        Y_list.append(target.astype(np.float32))

    if len(X_list) == 0:
        return np.zeros((0,)), np.zeros((0,))

    X = np.array(X_list)
    Y = np.array(Y_list)
    if Y.ndim == 2:
        Y = Y[:, :, None]

    return torch.FloatTensor(X), torch.FloatTensor(Y)

def normalize_adj(adj_np):
    adj_t = torch.FloatTensor(adj_np)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt)

def main():
    print("--- Starting Evaluation ---")

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    # Load Data
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    node_features = data_pack['node_features']
    adj_geo = data_pack['adj_geo']
    adj_conflict = data_pack['adj_conflict']

    # Normalize Adjacency
    norm_adj_geo = normalize_adj(adj_geo)
    norm_adj_conflict = normalize_adj(adj_conflict)
    norm_adj_list = [norm_adj_geo, norm_adj_conflict]

    # Prepare Dataset
    print(f"Preparing dataset from features shape: {node_features.shape}")
    X, Y = prepare_dataset(node_features)

    # Split (Last 20% for Validation)
    split_idx = int(len(X) * 0.8)
    X_val = X[split_idx:]
    Y_val = Y[split_idx:]

    print(f"Validation Set: {X_val.shape[0]} samples")

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_nodes = node_features.shape[0]
    model = STGCN(num_nodes=num_nodes, in_channels=26, time_steps=HISTORY_WINDOW, num_classes=1, num_graphs=len(norm_adj_list)).to(device)

    # Load Weights
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    norm_adj_list = [a.to(device) for a in norm_adj_list]
    model.eval()

    # Evaluation Loop
    mse_list = []
    rmse_list = []
    mae_list = []
    p5_list = []
    latencies = []

    dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss() # Standard MSE for reporting

    print("Running inference...")
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            start_time = time.time()
            output = model(batch_x, norm_adj_list)
            end_time = time.time()

            latencies.append((end_time - start_time) / batch_x.shape[0]) # Seconds per sample

            # Metrics
            mse = criterion(output, batch_y).item()
            mae = torch.mean(torch.abs(output - batch_y)).item()
            p5 = precision_at_k(output, batch_y, k=5)

            mse_list.append(mse)
            rmse_list.append(np.sqrt(mse))
            mae_list.append(mae)
            p5_list.append(p5)

    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_p5 = np.mean(p5_list)
    avg_latency_ms = np.mean(latencies) * 1000

    results = (
        f"--- Evaluation Results ---\n"
        f"Model: {MODEL_PATH}\n"
        f"Validation Samples: {len(X_val)}\n"
        f"MSE: {avg_mse:.4f}\n"
        f"RMSE: {avg_rmse:.4f}\n"
        f"MAE: {avg_mae:.4f}\n"
        f"Precision@5: {avg_p5:.4f}\n"
        f"Avg Latency per Sample: {avg_latency_ms:.2f} ms\n"
    )

    print(results)

    with open('ANALYSIS_RESULTS.txt', 'w') as f:
        f.write(results)

if __name__ == "__main__":
    main()

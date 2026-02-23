import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pandas as pd

# Adicionar raiz ao path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.architectures import TemperatureExpertGAT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def evaluate(model, X, y, adj_list):
    model.eval()
    prec_10_list = []
    prec_20_list = []
    losses = []

    with torch.no_grad():
        for vx, vy in zip(X, y):
            vpred = model(vx.to(DEVICE), adj_list).squeeze()
            vtrue = vy.squeeze().to(DEVICE)

            # Loss calculation (simplified for eval)
            loss = F.smooth_l1_loss(vpred, vtrue).item()
            losses.append(loss)

            vpred_np = vpred.cpu().numpy()
            vtrue_np = vtrue.cpu().numpy()

            if vtrue_np.sum() == 0: continue

            k_true_20 = min(20, len(vtrue_np))
            top_20_true = set(np.argsort(-vtrue_np)[:k_true_20])

            k_true_10 = min(10, len(vtrue_np))
            top_10_true = set(np.argsort(-vtrue_np)[:k_true_10])

            top_10_pred = set(np.argsort(-vpred_np)[:10])
            top_20_pred = set(np.argsort(-vpred_np)[:20])

            p10 = len(top_10_pred & top_10_true) / 10
            p20 = len(top_20_pred & top_20_true) / 20

            prec_10_list.append(p10)
            prec_20_list.append(p20)

    return np.mean(losses), np.mean(prec_10_list), np.mean(prec_20_list)

def main():
    print("🔬 INITIALIZING OVERFITTING DIAGNOSTIC...")

    # 1. Load Data
    path = os.path.join(ROOT, 'data', 'processed', 'processed_fortaleza.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)

    features = data['node_features']
    dates = pd.to_datetime(data['dates'])
    nodes_gdf = data['nodes_gdf']

    # 2. Apply SAME Filters as Training (Top 30, 2024-2026, Daily > 3)

    # Spatial Filter: Top 30
    total_cvli_per_node = features[:, :, 0].sum(axis=1)
    top_k_indices = np.argsort(-total_cvli_per_node)[:30]
    features = features[top_k_indices, :, :]
    nodes_gdf = nodes_gdf.iloc[top_k_indices].reset_index(drop=True)

    # Recalculate Adjacency
    coords = np.array(list(zip(nodes_gdf.geometry.x, nodes_gdf.geometry.y)))
    from scipy.spatial.distance import cdist
    adj_geo = (cdist(coords, coords) <= 3000).astype(float)
    adj_conf = np.eye(len(nodes_gdf))

    adj_geo_t = torch.tensor(normalize_adj(adj_geo), dtype=torch.float32).to(DEVICE)
    adj_conf_t = torch.tensor(normalize_adj(adj_conf), dtype=torch.float32).to(DEVICE)

    # Temporal Filter: 2024-2026
    mask_date = (dates >= pd.Timestamp('2024-01-01')) & (dates <= pd.Timestamp('2026-12-31'))
    features = features[:, mask_date, :]
    dates = dates[mask_date]

    # Daily Intensity Filter: > 3
    daily_sums = features[:, :, 0].sum(axis=0)
    mask_hot = daily_sums > 3
    features = features[:, mask_hot, :]

    print(f"📊 Data Stats: {len(nodes_gdf)} Nodes | {features.shape[1]} TimeSteps (Hot Days)")

    # 3. Prepare Tensors (Same logic as training)
    WINDOW, PREDICT_HORIZON = 30, 7
    N, T_total, C = features.shape

    X_list, y_list = [], []
    adj_dense = torch.tensor(adj_geo, dtype=torch.float32)
    features_norm = features.copy()
    for c in range(C):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features_norm[:, :, c] - m) / s

    for t in range(WINDOW, T_total - PREDICT_HORIZON):
        x_t = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
        y_target = y_raw + (0.3 * torch.matmul(adj_dense, y_raw))
        if y_target.max() > 0: y_target = y_target / y_target.max()
        X_list.append(x_t)
        y_list.append(y_target.unsqueeze(0))

    # 4. Split (Same 80/20 split)
    split = int(len(X_list) * 0.8)
    train_X, train_y = X_list[:split], y_list[:split]
    val_X, val_y = X_list[split:], y_list[split:]

    print(f"📉 Split: {len(train_X)} Train samples | {len(val_X)} Val samples")

    # 5. Load Model
    model_path = os.path.join(ROOT, 'models', 'test', 'ranking', 'fortaleza_expert_universal.pth')
    model = TemperatureExpertGAT(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=0.25).to(DEVICE)
    # Fix for PyTorch 2.6+ weights_only default
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"🤖 Model Loaded. Best Val Metric from Checkpoint: {checkpoint.get('p20', checkpoint.get('recall', 0))*100:.1f}%")

    # 6. Evaluate
    train_loss, train_p10, train_p20 = evaluate(model, train_X, train_y, [adj_geo_t, adj_conf_t])
    val_loss, val_p10, val_p20 = evaluate(model, val_X, val_y, [adj_geo_t, adj_conf_t])

    print("\n" + "="*40)
    print(f"🔍 OVERFITTING REPORT")
    print("="*40)
    print(f"TRAIN Set -> Loss: {train_loss:.4f} | P@10: {train_p10*100:.1f}% | P@20: {train_p20*100:.1f}%")
    print(f"VAL   Set -> Loss: {val_loss:.4f} | P@10: {val_p10*100:.1f}% | P@20: {val_p20*100:.1f}%")
    print("-" * 40)

    gap = train_p20 - val_p20
    print(f"⚠️  Generalization Gap (P@20): {gap*100:.1f}%")

    if gap > 0.15:
        print("❌ DIAGNOSIS: SIGNIFICANT OVERFITTING DETECTED (>15%)")
    elif gap < -0.05:
        print("❓ DIAGNOSIS: UNDERFITTING OR DATA LEAKAGE/SHIFT")
    else:
        print("✅ DIAGNOSIS: GOOD GENERALIZATION (Balanced Model)")

if __name__ == "__main__":
    main()

"""
Quick model evaluation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import numpy as np
from src.model import STGCN

DATA_FILE = 'data/processed/processed_graph_data.pkl'
MODEL_PATH = 'models/stgcn_model_v2.pth'

# Load
with open(DATA_FILE, 'rb') as f:
    d = pickle.load(f)

nf = d['node_features']  # (319, 1491, 8)

# Use adjacency from pickle or create identity
adj = d.get('adj_geo', np.eye(319))

print(f"Data: {nf.shape}, Adjacency: {adj.shape}")

# Quick windows
W = 30
X, Y = [], []
for t in range(nf.shape[1] - W):
    X.append(nf[:, t:t+W, :])
    Y.append(nf[:, t+W, :2])

X = np.transpose(np.array(X), (0, 3, 1, 2))  # (batch, features, nodes, time)
Y = np.array(Y)

split = int(0.8 * len(X))
X_val = torch.FloatTensor(X[split:])
Y_val = torch.FloatTensor(Y[split:])

print(f"Val set: X={X_val.shape}, Y={Y_val.shape}")

# Load model
device = torch.device('cpu')
model = STGCN(num_nodes=319, in_channels=8, time_steps=W).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Predict
with torch.no_grad():
    preds = model(X_val.to(device)).cpu().numpy()

print(f"Predictions: {preds.shape}")

# P@5
def p_at_5(Y_true, Y_pred):
    n_nodes = Y_true.shape[1]
    top_k = max(1, int(n_nodes * 0.05))
    
    ps = []
    for i in range(len(Y_true)):
        true_pos = np.where(Y_true[i].sum(axis=1) > 0)[0]  # nodes with any crime
        if len(true_pos) == 0:
            continue
        
        pred_top = np.argsort(Y_pred[i].mean(axis=1))[-top_k:]
        hits = len(np.intersect1d(true_pos, pred_top))
        ps.append(hits / len(true_pos))
    
    return np.mean(ps) if ps else 0

p = p_at_5(Y_val.numpy(), preds)

print(f"\n{'='*60}")
print(f"P@5 (current):  {p:.4f} ({p*100:.2f}%)")
print(f"P@5 (baseline): 0.1489 (14.89%)")
print(f"Difference:     {(p-0.1489)*100:+.2f}pp")
print(f"{'='*60}")

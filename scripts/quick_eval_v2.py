"""
Avaliação rápida do modelo ST-GCN v2 com 8 canais e 30 dias
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import numpy as np
from src.model import STGCN
from numpy.lib.stride_tricks import sliding_window_view

HISTORY_WINDOW = 30
MODEL_PATH = 'models/stgcn_model_v2.pth'
DATA_FILE = 'data/processed/processed_graph_data.pkl'

def prepare_dataset(node_features):
    windows = sliding_window_view(node_features, HISTORY_WINDOW, axis=1)
    X = windows[:, :-1, :, :]
    target_data = node_features[:, HISTORY_WINDOW:, 0:1]
    X = X.transpose(1, 2, 0, 3)
    Y = target_data.transpose(1, 0, 2)
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)

def precision_at_k(pred, target, k=5):
    batch_size = pred.shape[0]
    p_k_sum = 0.0
    valid = 0

    for i in range(batch_size):
        p = pred[i, :, 0].cpu().numpy()
        t = target[i, :, 0].cpu().numpy()

        if t.max() == 0:
            continue
        
        valid += 1
        _, true_top_k = torch.topk(torch.FloatTensor(t), min(k, len(t)))
        _, pred_top_k = torch.topk(torch.FloatTensor(p), min(k, len(p)))
        
        hits = len(set(true_top_k.numpy()) & set(pred_top_k.numpy()))
        p_k_sum += (hits / min(k, (t > 0).sum()))

    return p_k_sum / max(1, valid)

print("Carregando dados...")
with open(DATA_FILE, 'rb') as f:
    data_pack = pickle.load(f)

node_features = data_pack['node_features']
adj_geo = data_pack['adj_geo']
adj_conflict = data_pack['adj_conflict']

def normalize_adj(adj_np):
    adj_t = torch.FloatTensor(adj_np)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt)

norm_adj_list = [normalize_adj(adj_geo), normalize_adj(adj_conflict)]

print("Preparando dados...")
X, Y = prepare_dataset(node_features)
split_idx = int(len(X) * 0.8)
X_val, Y_val = X[split_idx:], Y[split_idx:]

print(f"Validação: {X_val.shape}")

device = torch.device('cpu')
num_nodes = node_features.shape[0]
num_features = node_features.shape[2]

print(f"Carregando modelo: {num_nodes} nodes, {num_features} features, {HISTORY_WINDOW} days...")
model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=HISTORY_WINDOW, num_classes=1, num_graphs=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

norm_adj_list = [a.to(device) for a in norm_adj_list]

print("Avaliando...")
val_p5 = 0.0
batch_size = 32
num_batches = (len(X_val) + batch_size - 1) // batch_size

with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_val))
        
        batch_x = torch.FloatTensor(X_val[start_idx:end_idx]).to(device)
        batch_y = torch.FloatTensor(Y_val[start_idx:end_idx]).to(device)
        
        output = model(batch_x, norm_adj_list)
        val_p5 += precision_at_k(output, batch_y, k=5)

avg_val_p5 = val_p5 / num_batches

print(f"\n{'='*50}")
print(f"ST-GCN v2 - 8 canais, 30 dias janela")
print(f"{'='*50}")
print(f"P@5: {avg_val_p5:.4f} ({avg_val_p5*100:.2f}%)")
print(f"{'='*50}\n")

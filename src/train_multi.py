import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# garantir import de src
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import STGCN

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataset(node_features, history_window, horizon_days=1):
    # node_features: (N, T, F)
    N, T, F = node_features.shape
    X, Y = [], []
    max_i = T - history_window - horizon_days + 1
    for i in range(max_i):
        window = node_features[:, i:i+history_window, :]
        target_window = node_features[:, i+history_window:i+history_window+horizon_days, :]
        # aggregate target over horizon (sum)
        target = np.sum(target_window, axis=1)  # (N, F)
        X.append(np.transpose(window, (2, 0, 1)))  # (F, N, history_window)
        Y.append(target)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def weighted_mse_loss(input, target, weights):
    sq_diff = (input - target) ** 2
    weighted_sq_diff = sq_diff * weights.view(1, 1, -1)
    return torch.mean(weighted_sq_diff)

def train_and_save(node_features, adj_matrix, history_window, horizon_days, epochs=2, batch_size=4, lr=1e-3):
    X, Y = prepare_dataset(node_features, history_window, horizon_days)
    if len(X) == 0:
        print('Dataset too small for given history/horizon')
        return None

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)), batch_size=batch_size, shuffle=False)

    # normalize adjacency
    adj_t = torch.FloatTensor(adj_matrix)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt).to(DEVICE)

    N = node_features.shape[0]
    in_channels = node_features.shape[2]

    model = STGCN(num_nodes=N, in_channels=in_channels, time_steps=history_window).to(DEVICE)
    loss_weights = torch.tensor([5.0, 1.0]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training model history={history_window} -> horizon={horizon_days} (samples={len(X)}) on {DEVICE}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx, norm_adj)
            loss = weighted_mse_loss(out, by, loss_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx, norm_adj)
                val_loss += float(weighted_mse_loss(out, by, loss_weights))

        print(f"Epoch {epoch+1}/{epochs} | TrainLoss: {train_loss/len(train_loader):.6f} | ValLoss: {val_loss/len(val_loader):.6f}")

    # save
    name = f'stgcn_{history_window}x{horizon_days}.pth'
    path = os.path.join(MODELS_DIR, name)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")
    return path

def main():
    print('Loading processed data...')
    with open(DATA_FILE, 'rb') as f:
        d = pickle.load(f)
    node_features = d['node_features']
    adj = d['adj_matrix']

    # Train requested variants: 30->7 (aggregate 7 days) and 30->1
    train_and_save(node_features, adj, history_window=30, horizon_days=7, epochs=3, batch_size=4)
    train_and_save(node_features, adj, history_window=30, horizon_days=1, epochs=3, batch_size=8)

if __name__ == '__main__':
    main()

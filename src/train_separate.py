import os
import sys
import time
import logging
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure we can import from src
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import STGCN

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def prepare_dataset(node_features, history_window, horizon, feature_idx):
    """
    node_features: (N, T, F)
    feature_idx: int, index of the feature to use.
    Returns X: (Samples, 1, N, Window), Y: (Samples, N) - assuming target is sum over horizon
    Wait, STGCN expects (Batch, In_Channels, Nodes, Time).
    My previous read of prepare_dataset in train_models.py:
    X.append(np.transpose(window, (2, 0, 1))) -> (F, N, T)
    """
    num_nodes, num_timesteps, num_features = node_features.shape

    # Extract specific feature
    # features: (N, T, 1)
    features = node_features[:, :, feature_idx:feature_idx+1]

    X, Y = [], []
    for i in range(0, num_timesteps - history_window - horizon + 1):
        # Window: (N, Window, 1)
        window = features[:, i:i+history_window, :]

        # Target: Sum over horizon for the specific feature
        # features[:, i+window:i+window+horizon, :] is (N, Horizon, 1)
        # Sum along axis 1 (time) -> (N, 1)
        target = np.sum(features[:, i+history_window:i+history_window+horizon, :], axis=1)

        # Transpose window to (C, N, T) for PyTorch
        # (N, T, C) -> (C, N, T)
        X.append(np.transpose(window, (2, 0, 1)).astype(np.float32))

        # Target shape: (N, 1) -> flatten to (N) if num_classes=1?
        # STGCN output is (Batch, N, num_classes).
        # If num_classes=1, target should be (Batch, N, 1).
        Y.append(target.astype(np.float32))

    return np.array(X), np.array(Y)

def weighted_mse_loss(input, target, weights=None):
    sq_diff = (input - target) ** 2
    if weights is not None:
        # weights should be applied based on value magnitude or class?
        # train_models.py used tensor([5.0, 1.0]) for 2 classes.
        # Here we have 1 class. We can skip weights or use a scalar.
        weighted_sq_diff = sq_diff * weights
        return torch.mean(weighted_sq_diff)
    return torch.mean(sq_diff)

def train_one(model, loader, optimizer, device, norm_adj, logger, epoch):
    model.train()
    total_loss = 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx, norm_adj)
        loss = weighted_mse_loss(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device, norm_adj):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx, norm_adj)
            loss = weighted_mse_loss(out, by)
            total_loss += loss.item()
    return total_loss / len(loader)

def run_training(feature_idx, history_window, horizon, out_path, epochs=30, batch_size=64, lr=1e-3):
    logger = setup_logger(f'train_{feature_idx}')
    logger.info(f'Starting training for feature {feature_idx}: window={history_window}, horizon={horizon}')

    data_file = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return

    with open(data_file, 'rb') as f:
        pack = pickle.load(f)

    node_features = pack['node_features']
    adj = pack['adj_matrix']

    # Filter dates >= 2024-01-01 if dates exist (following train_models.py logic)
    dates = pack.get('dates')
    if dates is not None:
        dates = list(dates)
        start_idx = 0
        for idx, d in enumerate(dates):
            if str(d) >= '2024-01-01':
                start_idx = idx
                break
        if start_idx > 0:
            node_features = node_features[:, start_idx:, :]
            logger.info(f'Subset data from 2024-01-01: new shape {node_features.shape}')

    X, Y = prepare_dataset(node_features, history_window, horizon, feature_idx)
    logger.info(f"Dataset created. X: {X.shape}, Y: {Y.shape}")

    if len(X) == 0:
        logger.error("No data to train.")
        return

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)), batch_size=batch_size, shuffle=False)

    # Prepare Adjacency
    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt).to(device)

    # Initialize Model
    # in_channels=1 because we extract only one feature
    # num_classes=1 because we predict one value (sum) per node
    num_nodes = node_features.shape[0]
    model = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=history_window, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    patience = 10
    no_improve = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(1, epochs+1):
        train_loss = train_one(model, train_loader, optimizer, device, norm_adj, logger, epoch)
        val_loss = validate(model, val_loader, device, norm_adj)

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), out_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Training finished. Best Val Loss: {best_val:.5f}. Model saved to {out_path}")

if __name__ == "__main__":
    # CVLI: Feature 0, Window 30, Horizon 7
    run_training(feature_idx=0, history_window=30, horizon=7, out_path=os.path.join(ROOT, 'models', 'stgcn_cvli.pth'))

    # CVP: Feature 1, Window 15, Horizon 1
    run_training(feature_idx=1, history_window=15, horizon=1, out_path=os.path.join(ROOT, 'models', 'stgcn_cvp.pth'))

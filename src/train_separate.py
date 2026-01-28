import os
import sys
import time
import logging
import pickle
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import STGCN
from torch.utils.data import Dataset

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S'))
        logger.addHandler(ch)
    return logger

def prepare_dataset(node_features, history_window, horizon, feature_idx, logger):
    num_nodes, num_timesteps, num_features = node_features.shape
    features = node_features[:, :, feature_idx:feature_idx+1]

    X, Y = [], []
    logger.info("Preparing dataset...")

    valid_range = num_timesteps - history_window - horizon + 1
    for i in range(0, valid_range):
        window = features[:, i:i+history_window, :]
        target = np.sum(features[:, i+history_window:i+history_window+horizon, :], axis=1)
        X.append(np.transpose(window, (2, 0, 1)).astype(np.float32))
        Y.append(target.astype(np.float32))

    logger.info("Converting to numpy arrays...")
    X_arr = np.array(X)
    Y_arr = np.array(Y)

    del X, Y
    gc.collect()
    return X_arr, Y_arr


class NodeFeatureWindowDataset(Dataset):
    """Dataset that reads `node_features.npy` with mmap and yields window/target pairs.

    Each item is (X, Y) where X has shape (channels, N, history) and Y has shape (N, 1).
    """
    def __init__(self, npy_path, history_window, horizon, feature_idx, start_idx=0):
        self.npy_path = npy_path
        self.history = history_window
        self.horizon = horizon
        self.feature_idx = feature_idx
        # memory-map the file (read-only)
        self.node_features = np.load(self.npy_path, mmap_mode='r')
        self.N, self.T, self.C = self.node_features.shape
        self.start_idx = start_idx
        self.valid_range = self.T - history_window - horizon + 1 - self.start_idx

    def __len__(self):
        return max(0, self.valid_range)

    def __getitem__(self, idx):
        i = idx + self.start_idx
        # slice features for selected feature_idx, keep last dim
        window = self.node_features[:, i:i+self.history, self.feature_idx:self.feature_idx+1]
        target = np.sum(self.node_features[:, i+self.history:i+self.history+self.horizon, self.feature_idx:self.feature_idx+1], axis=1)
        X = np.transpose(window, (2, 0, 1)).astype(np.float32)
        Y = target.astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(Y)

def weighted_mse_loss(input, target):
    return torch.mean((input - target) ** 2)

def train_one(model, loader, optimizer, device, norm_adj, logger):
    model.train()
    total_loss = 0.0
    for i, (bx, by) in enumerate(loader):
        if i == 0:
            logger.info(f"Batch shapes: bx={tuple(bx.shape)}, by={tuple(by.shape)}")
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx, norm_adj)
        loss = weighted_mse_loss(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        del bx, by, out, loss
    return total_loss / len(loader)

def run_training(feature_idx, history_window, horizon, out_path, epochs=10, batch_size=32, lr=1e-3):
    logger = setup_logger(f'train_{feature_idx}')
    logger.info(f'Starting training: window={history_window}, horizon={horizon}, batch={batch_size}')

    data_file = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    pack = {}
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            pack = pickle.load(f)
    else:
        # try to build minimal pack from graph_data directory (chunks after git strip)
        graph_dir = os.path.join(ROOT, 'data', 'processed', 'graph_data')
        if os.path.exists(graph_dir):
            nf_np = os.path.join(graph_dir, 'node_features.npy')
            adj_np = os.path.join(graph_dir, 'adj_matrix.npy')
            dates_pkl = os.path.join(graph_dir, 'dates.pkl')
            if os.path.exists(nf_np):
                pack['node_features'] = None
                # store path in node_features variable later
            if os.path.exists(adj_np):
                try:
                    pack['adj_matrix'] = np.load(adj_np, allow_pickle=True)
                except Exception:
                    pack['adj_matrix'] = None
            if os.path.exists(dates_pkl):
                try:
                    with open(dates_pkl, 'rb') as fh:
                        pack['dates'] = pickle.load(fh)
                except Exception:
                    pack['dates'] = None
        else:
            raise FileNotFoundError(f"processed_graph_data.pkl not found and graph_data directory missing: {graph_dir}")

    # prefer to load node_features via memmap from `data/processed/graph_data/node_features.npy`
    node_features = None
    adj = None
    dates = None
    graph_nf = os.path.join(os.path.dirname(data_file), 'graph_data', 'node_features.npy')
    if os.path.exists(graph_nf):
        node_features = graph_nf
    else:
        node_features = pack.get('node_features')
    adj = pack.get('adj_matrix')
    dates = pack.get('dates')

    if dates is not None and not isinstance(node_features, str):
        dates = list(dates)
        start_idx = 0
        for idx, d in enumerate(dates):
            if str(d) >= '2025-01-01':
                start_idx = idx
                break
        if start_idx > 0:
            node_features = node_features[:, start_idx:, :]

    del pack
    gc.collect()

    # If node_features is a path (npy memmap), use streaming Dataset to avoid large memory usage
    if isinstance(node_features, str) and os.path.exists(node_features):
        dataset = NodeFeatureWindowDataset(node_features, history_window, horizon, feature_idx)
        logger.info(f"Using memmap dataset from {node_features}, samples={len(dataset)}")
        if len(dataset) == 0:
            return
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        X, Y = prepare_dataset(node_features, history_window, horizon, feature_idx, logger)
        logger.info(f"Dataset: X={X.shape}, Y={Y.shape}")
        if len(X) == 0:
            return
        train_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        del X, Y
        gc.collect()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt).to(device)
    # model expects a list of adjacency matrices (one per graph); replicate if single
    norm_adj = [norm_adj, norm_adj]

    if isinstance(node_features, str) and os.path.exists(node_features):
        num_nodes = dataset.N
    else:
        num_nodes = node_features.shape[0]
    model = STGCN(num_nodes=num_nodes, in_channels=1, time_steps=history_window, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss = train_one(model, train_loader, optimizer, device, norm_adj, logger)
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.5f} | Time: {elapsed:.1f}s")
        torch.save(model.state_dict(), out_path)

if __name__ == "__main__":
    # CVLI: Feature 0, Window 90, Horizon 7
    run_training(feature_idx=0, history_window=90, horizon=7, out_path=os.path.join(ROOT, 'models', 'stgcn_cvli.pth'), batch_size=4, epochs=5)

    # CVP: Feature 1, Window 30, Horizon 1
    # run_training(feature_idx=1, history_window=30, horizon=1, out_path=os.path.join(ROOT, 'models', 'stgcn_cvp.pth'), batch_size=32, epochs=5)

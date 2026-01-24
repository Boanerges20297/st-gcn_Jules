"""
Script de treino otimizado com torch.compile() para CPU.
Usa compilação JIT para acelerar operações (30-50% mais rápido).
"""
import os
import sys
import time
import logging
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import STGCN


def setup_logger():
    logger = logging.getLogger('train_compiled')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S')
    ch.setFormatter(fmt)
    logger.handlers = []
    logger.addHandler(ch)
    return logger


def prepare_dataset(node_features, history_window, horizon):
    num_nodes, num_timesteps, num_features = node_features.shape
    X, Y = [], []
    for i in range(0, num_timesteps - history_window - horizon + 1):
        window = node_features[:, i:i+history_window, :]
        target = np.sum(node_features[:, i+history_window:i+history_window+horizon, :], axis=1)
        X.append(np.transpose(window, (2, 0, 1)).astype(np.float32))
        Y.append(target.astype(np.float32))
    return np.array(X), np.array(Y)


def weighted_mse_loss(input, target, weights):
    sq_diff = (input - target) ** 2
    weighted_sq_diff = sq_diff * weights.view(1, 1, -1)
    return torch.mean(weighted_sq_diff)


def train_one(model, loader, optimizer, device, norm_adj, loss_weights, logger, epoch, print_every=10):
    model.train()
    total_loss = 0.0
    batch_times = []
    
    for batch_idx, (bx, by) in enumerate(loader, 1):
        t0 = time.time()
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        
        out = model(bx, norm_adj)
        loss = weighted_mse_loss(out, by, loss_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_times.append(time.time() - t0)
        
        if batch_idx % print_every == 0 or batch_idx == len(loader):
            avg_bt = sum(batch_times[-print_every:]) / len(batch_times[-print_every:])
            batches_left = len(loader) - batch_idx
            eta = batches_left * avg_bt
            logger.info(f'Epoch {epoch} Batch {batch_idx}/{len(loader)} loss={loss.item():.6f} ETA={eta:.0f}s')
    
    return total_loss / len(loader)


def validate(model, loader, device, norm_adj, loss_weights):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx, norm_adj)
            loss = weighted_mse_loss(out, by, loss_weights)
            total_loss += loss.item()
    return total_loss / len(loader)


def run_training(history_window=30, horizon=1, epochs=50, batch_size=512, lr=1e-3, out_path=None):
    logger = setup_logger()
    logger.info(f'Starting training: history_window={history_window}, horizon={horizon}, epochs={epochs}')

    data_file = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    with open(data_file, 'rb') as f:
        pack = pickle.load(f)

    node_features = pack['node_features']
    adj = pack['adj_matrix']
    dates = pack.get('dates')

    # Subset 2024+
    if dates is not None:
        dates = list(dates)
        start_idx = 0
        for idx, d in enumerate(dates):
            if str(d) >= '2024-01-01':
                start_idx = idx
                break
        node_features = node_features[:, start_idx:, :]
        logger.info(f'Subsetting from 2024-01-01. New timesteps: {node_features.shape[1]}')

    N, T, F = node_features.shape
    logger.info(f'Data -> nodes: {N}, timesteps: {T}, features: {F}')

    X, Y = prepare_dataset(node_features, history_window, horizon)
    if len(X) == 0:
        raise RuntimeError('Not enough timesteps')

    split = int(0.7 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    logger.info(f'Samples -> train: {len(X_train)}, val: {len(X_val)}')

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), 
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)), 
                            batch_size=batch_size, shuffle=False)

    device = torch.device('cpu')
    torch.set_num_threads(8)
    logger.info(f'Using CPU with {torch.get_num_threads()} threads + torch.compile() optimization')

    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt).to(device)

    model = STGCN(num_nodes=N, in_channels=F, time_steps=history_window).to(device)
    loss_weights = torch.tensor([5.0, 1.0], dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Compilar modelo com torch.compile() para otimização JIT (CPU)
    try:
        model = torch.compile(model, backend='inductor')
        logger.info('Model compiled with torch.compile()')
    except Exception as e:
        logger.warning(f'torch.compile() não disponível: {e}. Usando modelo padrão.')

    best_val = float('inf')
    patience = 15
    no_improve_count = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss = train_one(model, train_loader, optimizer, device, norm_adj, loss_weights, logger, epoch)
        val_loss = validate(model, val_loader, device, norm_adj, loss_weights)
        elapsed = time.time() - t0
        logger.info(f'Epoch {epoch}/{epochs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={elapsed:.1f}s')

        if val_loss < best_val:
            best_val = val_loss
            no_improve_count = 0
            # Se compilado, salvar o modelo compilado
            if hasattr(model, '_orig_mod'):
                torch.save(model._orig_mod.state_dict(), out_path)
            else:
                torch.save(model.state_dict(), out_path)
            logger.info(f'Checkpoint saved: {out_path}')
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f'Early stopping: val_loss not improved for {patience} epochs')
                break


if __name__ == '__main__':
    logger = setup_logger()
    logger.info('=== Treino com torch.compile() (CPU otimizado) ===')
    
    out1 = os.path.join(ROOT, 'models', 'stgcn_30_1.pth')
    out7 = os.path.join(ROOT, 'models', 'stgcn_30_7.pth')
    
    try:
        t_start = time.time()
        run_training(history_window=30, horizon=1, epochs=50, batch_size=512, lr=1e-3, out_path=out1)
        run_training(history_window=30, horizon=7, epochs=50, batch_size=512, lr=1e-3, out_path=out7)
        elapsed = (time.time() - t_start) / 60
        logger.info(f'✓ Treino completado! Total: {elapsed:.1f} min')
    except Exception as e:
        logger.error(f'Erro: {e}', exc_info=True)

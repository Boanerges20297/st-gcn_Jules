import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
from src.model import STGCN
from torch.utils.data import DataLoader, TensorDataset

DATA_FILE = 'data/processed_graph_data.pkl'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'stgcn_model.pth')
HISTORY_WINDOW = 7
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001

LOSS_WEIGHTS = torch.tensor([5.0, 1.0])


def get_logger():
    logger = logging.getLogger('train')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def prepare_dataset(node_features):
    # Optimized using sliding_window_view to avoid loop and list appending

    # Create windows along the time axis (axis 1)
    # shape: (num_nodes, num_windows, num_features, HISTORY_WINDOW)
    windows = sliding_window_view(node_features, HISTORY_WINDOW, axis=1)

    # Discard the last window because we need a target for each window,
    # and the last window's target would be out of bounds.
    X = windows[:, :-1, :, :]

    # Transpose to match model input shape: (num_samples, num_features, num_nodes, HISTORY_WINDOW)
    # Current shape: (N, S, F, H) -> Transpose (1, 2, 0, 3)
    X = X.transpose(1, 2, 0, 3)

    # Targets correspond to the timestep immediately following each window
    # Shape: (num_nodes, num_samples, num_features)
    Y = node_features[:, HISTORY_WINDOW:, :]

    # Transpose to (num_samples, num_nodes, num_features)
    Y = Y.transpose(1, 0, 2)

    return X, Y

def weighted_mse_loss(input, target, weights):
    sq_diff = (input - target) ** 2
    weighted_sq_diff = sq_diff * weights.view(1, 1, -1)
    return torch.mean(weighted_sq_diff)

def main():
    logger = get_logger()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    logger.info("Carregando dados...")
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    node_features = data_pack['node_features']
    adj_matrix = data_pack['adj_matrix']
    
    adj_tensor = torch.FloatTensor(adj_matrix)
    rowsum = adj_tensor.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt)
    
    logger.info("Criando janelas temporais...")
    X, Y = prepare_dataset(node_features)
    
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    logger.info(f"Treino: {X_train.shape}, Validação: {X_val.shape}")
    
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")

    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]
    
    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=HISTORY_WINDOW).to(device)
    norm_adj = norm_adj.to(device)
    loss_weights = LOSS_WEIGHTS.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info("Iniciando treinamento...")
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        batch_times = []
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
            t0 = time.time()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, norm_adj)
            loss = weighted_mse_loss(output, batch_y, loss_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bt = time.time() - t0
            batch_times.append(bt)
            # log every 10 batches
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                avg_bt = sum(batch_times)/len(batch_times)
                batches_left = len(train_loader) - batch_idx
                eta = batches_left * avg_bt
                logger.info(f'Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} - batch_loss={loss.item():.6f} ETA={eta:.1f}s')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader, 1):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x, norm_adj)
                loss = weighted_mse_loss(output, batch_y, loss_weights)
                val_loss += loss.item()

        epoch_time = time.time() - epoch_start
        train_avg = train_loss / len(train_loader) if len(train_loader) else 0
        val_avg = val_loss / len(val_loader) if len(val_loader) else 0
        logger.info(f"Epoch {epoch+1}/{EPOCHS} completed in {epoch_time:.1f}s | Train Loss: {train_avg:.6f} | Val Loss: {val_avg:.6f}")
        # save per-epoch checkpoint
        epoch_path = os.path.join(MODEL_DIR, f'stgcn_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_path)
        logger.info(f'Checkpoint saved: {epoch_path}')

    # final save
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Modelo salvo em {MODEL_PATH}")

if __name__ == "__main__":
    main()

import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
from src.model import STGCN
from torch.utils.data import DataLoader, Dataset, TensorDataset

DATA_FILE = 'data/processed/processed_graph_data.pkl'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'stgcn_model.pth')
HISTORY_WINDOW = 7
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
GAMMA = 2.0

def get_logger():
    logger = logging.getLogger('train')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

class BalancedWindowDataset(Dataset):
    def __init__(self, X, Y):
        # Ensure writable copies
        if isinstance(X, np.ndarray):
            X = X.copy()
        if isinstance(Y, np.ndarray):
            Y = Y.copy()

        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.num_samples = self.X.shape[0]

        # Identify positive windows (any CVLI in the target day)
        total_events = self.Y.sum(dim=(1, 2))
        self.positive_indices = torch.where(total_events > 0)[0]
        self.all_indices = torch.arange(self.num_samples)

        print(f"Total Samples: {self.num_samples}")
        print(f"Positive Samples: {len(self.positive_indices)}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.rand(1).item() < 0.5 and len(self.positive_indices) > 0:
            rand_pos_idx = self.positive_indices[torch.randint(0, len(self.positive_indices), (1,)).item()]
            return self.X[rand_pos_idx], self.Y[rand_pos_idx]
        else:
            rand_idx = self.all_indices[torch.randint(0, self.num_samples, (1,)).item()]
            return self.X[rand_idx], self.Y[rand_idx]

class WeightedFocalMSELoss(nn.Module):
    def __init__(self, weight_zero=1.0, weight_hotspot=500.0, gamma=2.0):
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

def prepare_dataset(node_features):
    windows = sliding_window_view(node_features, HISTORY_WINDOW, axis=1)

    X = windows[:, :-1, :, :] # (Nodes, Samples, Features, WindowSize)
    target_data = node_features[:, HISTORY_WINDOW:, 0:1] # (Nodes, Samples, 1)

    X = X.transpose(1, 2, 0, 3) # (Samples, Features, Nodes, WindowSize)
    Y = target_data.transpose(1, 0, 2) # (Samples, Nodes, 1)

    # Ensure numpy arrays are contiguous and writable
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)

    return X, Y

def precision_at_k(pred, target, k=5):
    batch_size = pred.shape[0]
    p_k_sum = 0.0

    for i in range(batch_size):
        p = pred[i, :, 0]
        t = target[i, :, 0]

        _, top_k_indices = torch.topk(p, k)

        hits = (t[top_k_indices] > 0).float().sum()
        p_k_sum += (hits / k).item()

    return p_k_sum / batch_size

def main():
    logger = get_logger()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    logger.info("Carregando dados...")
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

    norm_adj_geo = normalize_adj(adj_geo)
    norm_adj_conflict = normalize_adj(adj_conflict)
    norm_adj_list = [norm_adj_geo, norm_adj_conflict]

    logger.info("Criando janelas temporais...")
    X, Y = prepare_dataset(node_features)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    logger.info(f"Treino: {X_train.shape}, Validação: {X_val.shape}")
    
    # Datasets
    train_dataset = BalancedWindowDataset(X_train, Y_train)
    # Ensure validation data is also writable/contiguous before TensorDataset
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val.copy()),
        torch.FloatTensor(Y_val.copy())
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")

    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]
    
    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=HISTORY_WINDOW, num_classes=1, num_graphs=len(norm_adj_list)).to(device)
    norm_adj_list = [a.to(device) for a in norm_adj_list]
    
    criterion = WeightedFocalMSELoss(weight_zero=1.0, weight_hotspot=500.0, gamma=GAMMA).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info("Iniciando treinamento...")
    best_p5 = 0.0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, norm_adj_list)
            loss = criterion(output, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_p5 = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x, norm_adj_list)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                val_p5 += precision_at_k(output, batch_y, k=5)

        train_avg_loss = train_loss / len(train_loader)
        val_avg_loss = val_loss / len(val_loader)
        val_avg_p5 = val_p5 / len(val_loader)

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | Val P@5: {val_avg_p5:.4f}")

        if val_avg_p5 > best_p5:
            best_p5 = val_avg_p5
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"  -> Novo melhor modelo salvo! (P@5: {best_p5:.4f})")
        elif epoch == 0:
             torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()

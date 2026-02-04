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
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import STGCN
from torch.utils.data import DataLoader, Dataset, TensorDataset

DATA_FILE = 'data/processed/processed_graph_data.pkl'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'stgcn_model_v2.pth')  # v2: 26 canais categóricos (one-hot)
HISTORY_WINDOW = 14  # Reduzido de 30: captura padrões mais recentes e relevantes
BATCH_SIZE = 64  # 1. Ajuste fino: 16-32 para gradientes mais estáveis
EPOCHS = 60  # 5. Ajuste fino: 50+ para convergência completa
LEARNING_RATE = 0.0002  # 2. Ajuste fino: Manter 0.0001 (dados em escala original)
GAMMA = 1.5  # 3. Ajuste fino: Testar 1.5-2.0 (amplificação moderada)
WEIGHT_DECAY = 1e-5

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
        return self.X[idx], self.Y[idx]

class WeightedFocalMSELoss(nn.Module):
    """Loss com foco em hotspots COM amplificação."""
    def __init__(self, weight_zero=1.0, weight_hotspot=20.0, gamma=1.5):
        super(WeightedFocalMSELoss, self).__init__()
        self.weight_zero = weight_zero
        self.weight_hotspot = weight_hotspot  # Amplificação: 30.0
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
    """Prepara dataset ORIGINAL - sem balanceamento."""
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
    """Precision@K: entre os top-k nós com MAIS eventos reais,
    quantos o modelo predisse corretamente?
    CORRIGIDO: Contar apenas dias com eventos
    """
    batch_size = pred.shape[0]
    p_k_sum = 0.0
    valid_days = 0  # Dias com eventos reais

    for i in range(batch_size):
        p = pred[i, :, 0].detach().cpu().numpy()
        t = target[i, :, 0].detach().cpu().numpy()

        # Top-K com MAIS eventos reais
        if t.max() == 0:
            # Se não há eventos reais neste dia, skip
            continue
        
        valid_days += 1
        num_events = (t > 0).sum()
        k_actual = min(k, num_events, len(t))
        
        if k_actual <= 0:
            continue
            
        _, true_top_k_indices = torch.topk(torch.FloatTensor(t), k_actual)
        true_top_k_indices = true_top_k_indices.numpy()
        
        # Quantas vezes o modelo predisse alta também para esses nós?
        pred_top_k = torch.topk(torch.FloatTensor(p), k_actual)[1].numpy()
        
        hits = len(set(true_top_k_indices) & set(pred_top_k))
        p_k_sum += (hits / k_actual)

    return p_k_sum / max(1, valid_days)  # Dividir apenas por dias com eventos

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
    
    model = STGCN(num_nodes=num_nodes, in_channels=26, time_steps=HISTORY_WINDOW, num_classes=1, num_graphs=len(norm_adj_list)).to(device)
    norm_adj_list = [a.to(device) for a in norm_adj_list]
    
    criterion = WeightedFocalMSELoss(weight_zero=1.0, weight_hotspot=20.0, gamma=GAMMA).to(device)  # 4. Ajuste fino: 15-25 para foco em eventos raros
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    logger.info("Iniciando treinamento...")
    best_p5 = 0.0
    patience = 15  # Early stopping: 15 epochs sem melhora
    patience_counter = 0

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
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | Val P@10: {val_avg_p5:.4f}")

        # Early stopping
        if val_avg_p5 > best_p5:
            best_p5 = val_avg_p5
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"  -> Novo melhor modelo salvo! (P@10: {best_p5:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  -> Early stopping! ({patience_counter} epochs sem melhora)")
                break
        
        scheduler.step(val_avg_p5)

if __name__ == "__main__":
    main()

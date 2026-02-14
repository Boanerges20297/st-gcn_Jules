import os
import sys
import time
import pickle
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from model_v4 import DeepSTGAT
except ImportError:
    from Phase4.model_v4 import DeepSTGAT

# Configurações: Foco no MOMENTUM ATUAL (Regime 2024-2026)
DATA_FILE = 'data/processed/processed_graph_data.pkl'
MODEL_DIR = 'models/phase5' 
MODEL_PATH = os.path.join(MODEL_DIR, 'best_stgat_v5_massive.pth')
LOG_FILE = 'Phase5_fortaleza_lab.log'
HISTORY_WINDOW = 30
BATCH_SIZE = 16 
EPOCHS = 100
MAX_LR = 0.005 

# Logging
log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger = logging.getLogger('Lab-Momentum')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False

class TimeWeightedDataset(Dataset):
    def __init__(self, node_features, history_window=30, horizon=7):
        self.features = torch.from_numpy(node_features).float()
        self.window = history_window
        self.horizon = horizon
        self.num_samples = node_features.shape[1] - self.window - self.horizon + 1
        
        # Sampler prioriza o PRESENTE (2024-2026)
        # O peso cresce quadraticamente com o tempo
        self.sample_weights = []
        for idx in range(self.num_samples):
            # Normaliza o tempo (0.0 no inicio, 1.0 no fim)
            time_factor = (idx / self.num_samples) ** 2 
            
            # Se tiver crime, multiplica o peso
            target = self.features[:, idx + self.window : idx + self.window + self.horizon, 0].sum()
            crime_factor = 5.0 if target > 0 else 1.0
            
            # Peso final combina Recência + Ocorrência
            self.sample_weights.append(0.1 + (time_factor * crime_factor))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.features[:, idx : idx + self.window, :]
        x = x.permute(2, 0, 1) 
        y = self.features[:, idx + self.window : idx + self.window + self.horizon, 0].sum(dim=1)
        
        # Retorna o indice temporal normalizado (0 a 1) para a Loss saber "quando" estamos
        time_idx = idx / self.num_samples
        return x, y.unsqueeze(-1), torch.tensor(time_idx, dtype=torch.float32)

class TemporalRegimeLoss(nn.Module):
    """
    Loss que diferencia a 'Guerra Passada' da 'Consolidação Atual'.
    Erros no presente custam muito mais caro.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, pred, target, time_idx):
        target_binary = (target > 0).float()
        
        # Calcula a Loss base (BCE)
        # Pos_weight alto para lidar com a esparsidade
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target_binary, 
            pos_weight=torch.tensor([50.0]).to(self.device), 
            reduction='none'
        )
        
        # Aplica o Peso Temporal (Regime Weight)
        # Passado (time_idx ~ 0) -> Peso 0.2 (Apenas contexto)
        # Presente (time_idx ~ 1) -> Peso 5.0 (Prioridade Máxima)
        time_weights = 0.2 + (time_idx ** 3) * 5.0 
        
        # Expande para o shape do batch [Batch, Nodes, 1]
        time_weights = time_weights.view(-1, 1, 1).expand_as(bce_loss)
        
        weighted_loss = bce_loss * time_weights
        return weighted_loss.mean()

def calculate_recall_at_10(pred, target):
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    valid_mask = target.max(dim=1)[0] > 0
    if not valid_mask.any(): return 0.0, 0
    p_v, t_v = pred[valid_mask], target[valid_mask]
    recall_sum = 0
    for i in range(p_v.shape[0]):
        top_10_idx = torch.topk(p_v[i], 10)[1]
        real_pos = (t_v[i] > 0).nonzero(as_tuple=True)[0]
        if len(real_pos) > 0:
            hits = len(set(top_10_idx.tolist()) & set(real_pos.tolist()))
            recall_sum += hits / len(real_pos)
    return recall_sum / p_v.shape[0], p_v.shape[0]

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = np.diag(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat)

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("--- INICIANDO TREINO DE REGIME (PRIORIDADE 2024-2026) ---")

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    node_features = data['node_features']
    # Não fazemos split fixo de tempo, pois queremos treinar com foco no fim
    # Usaremos os últimos 30 dias para validação "hold-out"
    train_features = node_features[:, :-30, :]
    val_features = node_features[:, -60:, :] # Valida nos últimos 2 meses reais
    
    train_ds = TimeWeightedDataset(train_features)
    # Sampler focado no tempo recente
    sampler = WeightedRandomSampler(train_ds.sample_weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    
    val_ds = TimeWeightedDataset(val_features)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    adj_list = [torch.from_numpy(normalize_adj(data['adj_geo'])).float().to(device), 
                torch.from_numpy(normalize_adj(data['adj_conflict'])).float().to(device)]

    model = DeepSTGAT(num_nodes=node_features.shape[0], in_channels=29, time_steps=HISTORY_WINDOW).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = TemporalRegimeLoss(device=device)

    best_recall = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, start_t = 0, time.time()
        for bx, by, t_idx in train_loader:
            bx, by, t_idx = bx.to(device), by.to(device), t_idx.to(device)
            optimizer.zero_grad()
            out = model(bx, adj_list)
            loss = criterion(out, by, t_idx)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        v_rec_sum, v_count = 0, 0
        with torch.no_grad():
            for bx, by, t_idx in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx, adj_list)
                rec, count = calculate_recall_at_10(out, by)
                v_rec_sum += rec * count
                v_count += count
        
        avg_recall = v_rec_sum / v_count if v_count > 0 else 0
        dt = time.time() - start_t
        logger.info(f"EP {epoch+1:03d} | Recall (Recente): {avg_recall:.2%} | Loss: {total_loss:.2f} | {dt:.1f}s/ep")
        
        if avg_recall > best_recall:
            best_recall = avg_recall
            torch.save({'model_state_dict': model.state_dict(), 'recall': best_recall}, MODEL_PATH)
            logger.info(f"  >>> NOVO RECORDE (REGIME ATUAL): {best_recall:.2%}")

if __name__ == "__main__":
    train()

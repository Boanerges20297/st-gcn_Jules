import os
import time
import pickle
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model_v4 import DeepSTGAT

# Configurações de Escala Massiva
DATA_FILE = 'data/processed/processed_graph_data.pkl'
MODEL_DIR = 'models/phase4'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_stgat_v4_massive.pth')
HISTORY_WINDOW = 30
HORIZON = 7
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('Phase5-Massive')

class LazyCrimeDataset(Dataset):
    """
    Dataset Otimizado para Grandes Volumes de RAM (48GB+).
    Carrega fatias de 1000 dias sob demanda.
    """
    def __init__(self, node_features, history_window=30, horizon=7):
        # node_features shape: (Nodes, Total_Days, Features)
        self.features = torch.from_numpy(node_features).float()
        self.window = history_window
        self.horizon = horizon
        self.num_nodes = node_features.shape[0]
        self.total_days = node_features.shape[1]
        
        # Número de janelas possíveis
        self.num_samples = self.total_days - self.window - self.horizon + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: (Nodes, Window, Features) -> Transpose to (Features, Nodes, Window)
        x = self.features[:, idx : idx + self.window, :]
        x = x.permute(2, 0, 1) 
        
        # Target: Soma de crimes no horizonte (canal 0)
        y = self.features[:, idx + self.window : idx + self.window + self.horizon, 0].sum(dim=1)
        return x, y.unsqueeze(-1) # (Nodes, 1)

class HybridRankingLoss(nn.Module):
    def __init__(self, hotspot_weight=35.0, device='cpu'):
        super().__init__()
        self.hotspot_weight = hotspot_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hotspot_weight]).to(device))

    def forward(self, pred, target):
        # Intensidade
        mse_loss = self.mse(pred, target)
        # Detecção de Hotspot
        target_binary = (target > 0).float()
        bce_loss = self.bce(pred, target_binary)
        return mse_loss + (bce_loss * 5.0)

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"--- INICIANDO PHASE 5: ESCALA MASSIVA ---")
    logger.info(f"Dispositivo: {device} | RAM Disponível: Alta")

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    node_features = data['node_features'] # (319, 1001, 26)
    
    # Split temporal (80% treino, 20% validação)
    total_days = node_features.shape[1]
    split_day = int(total_days * 0.8)
    
    train_features = node_features[:, :split_day, :]
    val_features = node_features[:, split_day - HISTORY_WINDOW :, :] # Overlap para janela
    
    train_ds = LazyCrimeDataset(train_features)
    val_ds = LazyCrimeDataset(val_features)
    
    # No Windows, num_workers > 0 pode ser instável em alguns setups, 
    # mas com 48GB e i5, 2 ou 4 costumam funcionar bem.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    adj_list = [torch.from_numpy(data['adj_geo']).float().to(device), 
                torch.from_numpy(data['adj_conflict']).float().to(device)]

    model = DeepSTGAT(num_nodes=node_features.shape[0], in_channels=26, time_steps=HISTORY_WINDOW).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = HybridRankingLoss(hotspot_weight=35.0, device=device).to(device)

    best_p10 = 0.0
    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            start_t = time.time()
            
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx, adj_list)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validação
            model.eval()
            val_p10 = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx, adj_list)
                    
                    # Cálculo de P@10 (Simplificado para velocidade)
                    for i in range(bx.shape[0]):
                        t = by[i].flatten()
                        p = out[i].flatten()
                        if t.max() > 0:
                            top_t = torch.topk(t, min(10, len(t)))[1]
                            top_p = torch.topk(p, min(10, len(p)))[1]
                            val_p10 += len(set(top_t.tolist()) & set(top_p.tolist())) / 10
            
            avg_p10 = val_p10 / len(val_ds)
            dt = time.time() - start_t
            logger.info(f"EP {epoch+1:03d} | P@10: {avg_p10:.4f} | Loss: {epoch_loss:.4f} | Tempo: {dt:.1f}s")
            
            if avg_p10 > best_p10:
                best_p10 = avg_p10
                torch.save(model.state_dict(), MODEL_PATH)
                logger.info(f"  >>> NOVO RECORDE SALVO: {best_p10:.4f}")

            scheduler.step(avg_p10)
            
    except Exception as e:
        logger.error(f"ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()

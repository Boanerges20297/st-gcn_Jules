import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import logging
import pandas as pd
import gc
from torch.utils.data import DataLoader, TensorDataset

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64
except ImportError:
    from src.core.architectures import DeepSTGAT_64

# Configuração de Log Robusta
log_file = os.path.join(ROOT_DIR, 'logs', 'training_ELITE_P10.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 90
PREDICT_HORIZON = 7
GAP = 7

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

class SpecialistTrainer:
    def __init__(self, region_key, epochs, lr, bs, dropout, rank_w):
        self.region_key = region_key
        self.epochs = epochs
        self.lr = lr
        self.bs = bs
        self.dropout = dropout
        self.rank_w = rank_w
        self.best_p10 = 0.0

    def train(self):
        logging.info(f"🚀 INICIANDO PADRÃO BRUTO: {self.region_key.upper()} | Device: {DEVICE}")
        
        path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        
        # Carregamento seguro: Forçar geopandas a estar no path se o pkl precisar
        try:
            import geopandas as gpd
        except ImportError:
            logging.warning("⚠️ Geopandas não encontrado, tentando carregar pkl mesmo assim...")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        nf = data['node_features'] # (N, T, 29)
        adj_geo_np = data['adj_geo']
        adj_conf_np = data['adj_conflict']
        N, T, C = nf.shape

        # Mover para device imediatamente
        adj_geo = torch.tensor(normalize_adj(adj_geo_np), dtype=torch.float32).to(DEVICE)
        adj_conf = torch.tensor(adj_conf_np, dtype=torch.float32).to(DEVICE)

        # Normalização de entrada (Z-Score)
        f_norm = nf.copy()
        for c in range(C):
            m, s = nf[:,:,c].mean(), nf[:,:,c].std() + 1e-5
            f_norm[:,:,c] = (nf[:,:,c] - m) / s

        X, Y = [], []
        for t in range(WINDOW, T - PREDICT_HORIZON):
            x = torch.tensor(f_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2,0,1)
            y_raw = torch.tensor(nf[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
            
            # Alvo Bruto Escalado (0 a 1 relativo ao batch para estabilidade)
            y_target = y_raw / (y_raw.max() + 1e-5) if y_raw.max() > 0 else y_raw
            
            X.append(x)
            Y.append(y_target)

        X, Y = torch.stack(X), torch.stack(Y)
        
        val_size = 60
        train_limit = len(X) - val_size - GAP
        
        x_train, y_train = X[:train_limit], Y[:train_limit]
        x_val, y_val = X[-val_size:], Y[-val_size:]

        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=self.bs, shuffle=True)
        
        model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=self.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=len(loader)*self.epochs)

        for epoch in range(self.epochs):
            model.train()
            for i, (bx, by) in enumerate(loader):
                batch_start = time.time()
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                
                pred = model(bx, [adj_geo, adj_conf]).squeeze(-1)
                
                # Loss de Impacto Bruto (Pesando 15x mais onde o crime real ocorreu)
                loss_reg = (F.smooth_l1_loss(pred, by, reduction='none') * (1.0 + by * 15.0)).mean()
                
                # Ranking Bruto
                _, top_idx = torch.topk(by, min(10, N), dim=1)
                top_scores = pred.gather(1, top_idx)
                loss_rank = F.relu(0.8 - (top_scores - pred.mean(dim=1, keepdim=True))).mean()
                
                loss = loss_reg + 30.0 * loss_rank
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # LOG DETALHADO POR BATCH
                if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                    elapsed = time.time() - batch_start
                    pred_np, by_np = pred.detach().cpu().numpy(), by.cpu().numpy()
                    p10_list = []
                    for k in range(len(by_np)):
                        if by_np[k].sum() > 0:
                            t_true = np.argsort(by_np[k])[::-1][:10]
                            t_pred = np.argsort(pred_np[k])[::-1][:10]
                            p10_list.append(len(set(t_true) & set(t_pred)) / 10.0)
                    
                    avg_p10 = np.mean(p10_list) if p10_list else 0.0
                    current_lr = optimizer.param_groups[0]['lr']
                    logging.info(f"   > E{epoch+1:02d} B{i+1:03d}/{len(loader)} | LR: {current_lr:.5f} | Loss: {loss.item():.4f} | P@10: {avg_p10*100:.1f}% | {elapsed:.2f}s")

            # Validação
            model.eval()
            with torch.no_grad():
                vp = model(x_val.to(DEVICE), [adj_geo, adj_conf]).squeeze(-1).cpu().numpy()
                vt = y_val.numpy()
                p10s = []
                for k in range(len(vt)):
                    if vt[k].sum() > 0:
                        it = np.argsort(vt[k])[::-1][:10]
                        ip = np.argsort(vp[k])[::-1][:10]
                        p10s.append(len(set(it) & set(ip)) / 10.0)
                
                m_p10 = np.mean(p10s) if p10s else 0.0
                logging.info(f"[{self.region_key.upper()}] E{epoch+1:02d} | Val P@10: {m_p10*100:.1f}%")
                
                if m_p10 > self.best_p10:
                    self.best_p10 = m_p10
                    torch.save({'model_state_dict': model.state_dict(), 'p10': m_p10}, f'models/active/{self.region_key}_model.pth')
                    logging.info(f"✨ RECORDE BRUTO: {m_p10*100:.1f}%")

def main():
    configs = [
        ('fortaleza', 120, 0.005, 32, 0.5, 30.0),
        ('rmf', 100, 0.005, 32, 0.5, 20.0),
        ('interior', 100, 0.005, 32, 0.5, 20.0)
    ]
    for k, e, lr, bs, dr, rw in configs:
        try:
            trainer = SpecialistTrainer(k, e, lr, bs, dr, rw)
            trainer.train()
        except Exception as e:
            logging.error(f"❌ ERRO CRÍTICO EM {k.upper()}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

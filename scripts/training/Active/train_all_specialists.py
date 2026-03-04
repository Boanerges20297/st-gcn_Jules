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
import random
from torch.utils.data import DataLoader, TensorDataset

# Adicionar raiz ao path para imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.core.architectures import DeepSTGAT_64
except ImportError:
    sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))
    from architectures import DeepSTGAT_64

# Configuração de Logging Unificado
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_all_specialists.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configurações Globais (Otimizadas para CPU/Intel Iris)
DEVICE = torch.device('cpu') # Forçar CPU já que não há GPU dedicada
WINDOW = 120
PREDICT_HORIZON = 7
GAP = PREDICT_HORIZON + 7 

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

class SpecialistTrainer:
    def __init__(self, region_key, epochs=30, lr=0.01, batch_size=16, dropout=0.3, ranking_weight=15.0):
        self.region_key = region_key
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.ranking_weight = ranking_weight
        self.best_p20 = 0.0
        
    def train(self):
        region_label = self.region_key.upper()
        logging.info("\n" + "="*80)
        logging.info(f"🚀 INICIANDO TREINAMENTO SEMANAL VIÁVEL: {region_label}")
        logging.info(f"📊 Config: LR={self.lr}, Batch={self.batch_size}, Épocas={self.epochs}")
        logging.info("="*80)
        
        data_path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        if not os.path.exists(data_path):
            logging.error(f"❌ Arquivo de dados não encontrado: {data_path}")
            return

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        features = data['node_features'] 
        dates = pd.to_datetime(data['dates'])
        N, T_total, C = features.shape
        
        # Matrizes de Adjacência em Tensores (CPU)
        adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32)
        adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32)
        adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)

        # Normalização Rápida
        features_norm = features.copy()
        for c in range(C):
            mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
            features_norm[:, :, c] = (features[:, :, c] - mean) / std

        # Preparação Vetorizada de Datasets
        X_all, y_all, weights_all = [], [], []
        
        # Pesos temporais pré-calculados
        df_temp = pd.DataFrame({'date': dates, 'crimes': features[:, :, 0].sum(axis=0)})
        avg_crimes = df_temp['crimes'].mean() + 1e-6
        month_weights = {m: max(0.8, min(1.3, val/avg_crimes)) for m, val in df_temp.groupby(df_temp['date'].dt.month)['crimes'].mean().items()}
        day_weights = {d: max(0.8, min(1.3, val/avg_crimes)) for d, val in df_temp.groupby(df_temp['date'].dt.dayofweek)['crimes'].mean().items()}

        logging.info(f"📦 Vetorizando janelas temporais para {region_label}...")
        for t in range(WINDOW, T_total - PREDICT_HORIZON):
            x_tensor = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1)
            y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
            
            # Suavização de Label (Pré-calculada)
            y_target = y_raw + (0.2 * torch.matmul(adj_dense, y_raw))
            if y_target.max() > 0: y_target = y_target / y_target.max()
            
            t_mult = month_weights.get(dates[t].month, 1.0) * day_weights.get(dates[t].dayofweek, 1.0)
            
            X_all.append(x_tensor)
            y_all.append(y_target)
            weights_all.append(t_mult)

        X_all = torch.stack(X_all)
        y_all = torch.stack(y_all)
        weights_all = torch.tensor(weights_all, dtype=torch.float32)

        # Divisão de Lastro (Final do dataset para validação)
        lastro_days = 60
        total_idx = len(X_all)
        train_limit = total_idx - lastro_days - GAP
        
        X_train, y_train, w_train = X_all[:train_limit], y_all[:train_limit], weights_all[:train_limit]
        X_val, y_val = X_all[total_idx-lastro_days:], y_all[total_idx-lastro_days:]
        
        # DataLoader real para eficiência de CPU
        train_ds = TensorDataset(X_train, y_train, w_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=self.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        logging.info(f"🎬 Iniciando Treino: {len(X_train)} amostras | Batch: {self.batch_size} | Device: {DEVICE}")

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            start_epoch = time.time()
            
            for i, (bx, by, bw) in enumerate(train_loader):
                batch_start = time.time()
                optimizer.zero_grad()
                pred = model(bx, [adj_geo, adj_conf]).squeeze(-1)
                
                # Loss Vetorizada (Regressão + Ranking Simplificado)
                loss_reg = (F.smooth_l1_loss(pred, by, reduction='none').mean(dim=1) * bw).mean()
                
                # Ranking (Top-K aproximado para velocidade)
                loss_rank = torch.tensor(0.0)
                if self.ranking_weight > 0:
                    _, top_idx = torch.topk(by, 20, dim=1)
                    loss_rank = F.relu(0.5 - (pred.gather(1, top_idx) - pred.mean(dim=1, keepdim=True))).mean()
                
                loss = loss_reg + self.ranking_weight * loss_rank
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Log detalhado por batch a cada 5 batches
                if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                    elapsed = time.time() - batch_start
                    progress = (i + 1) / len(train_loader) * 100
                    logging.info(f"   > Batch {i+1}/{len(train_loader)} ({progress:.1f}%) | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            
            scheduler.step()
            epoch_time = time.time() - start_epoch
            
            # Avaliação de Precisão Semanal
            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                p20_list = []
                with torch.no_grad():
                    val_pred = model(X_val, [adj_geo, adj_conf]).squeeze(-1)
                    for i in range(len(y_val)):
                        y_true_np, y_pred_np = y_val[i].numpy(), val_pred[i].numpy()
                        if np.sum(y_true_np) > 0:
                            t_true = np.argsort(y_true_np)[-20:]
                            t_pred = np.argsort(y_pred_np)[-20:]
                            p20_list.append(len(set(t_true) & set(t_pred)) / 20.0)
                
                current_p20 = np.mean(p20_list or [0])
                logging.info(f"✅ [{region_label}] E{epoch+1:02d} | Avg Loss: {total_loss/len(train_loader):.4f} | Val P@20: {current_p20*100:.1f}% | Time: {epoch_time:.1f}s")
                
                if current_p20 > self.best_p20:
                    self.best_p20 = current_p20
                    save_path = os.path.join(ROOT_DIR, 'models', 'active', f'{self.region_key}_model_active.pth')
                    torch.save({'model_state_dict': model.state_dict(), 'p20': current_p20}, save_path)
                    logging.info(f"🏆 NOVO RECORDE {region_label}: P@20={current_p20*100:.1f}%")

def main():
    os.makedirs(os.path.join(ROOT_DIR, 'models', 'active'), exist_ok=True)
    # Configuração Padrão Semanal Viável (Otimizada para CPU)
    tasks = [
        ('fortaleza', 30, 0.01, 16, 0.3, 10.0),
        ('rmf', 30, 0.01, 16, 0.3, 10.0),
        ('interior', 30, 0.01, 16, 0.3, 10.0)
    ]
    for key, epochs, lr, bs, drop, rank_w in tasks:
        trainer = SpecialistTrainer(key, epochs, lr, bs, drop, rank_w)
        trainer.train()
    
    logging.info("\n✅ TREINAMENTO SEMANAL CONCLUÍDO COM SUCESSO PARA TODOS OS ESPECIALISTAS.")

if __name__ == "__main__":
    main()

def main():
    os.makedirs(os.path.join(ROOT_DIR, 'models', 'active'), exist_ok=True)
    # Configurações Elite T32 ISM
    tasks = [
        ('fortaleza', 80, 0.05, 32, 0.4, 20.0),
        ('rmf', 80, 0.05, 32, 0.4, 25.0),
        ('interior', 80, 0.05, 32, 0.3, 20.0)
    ]
    for key, epochs, lr, accum, drop, rank_w in tasks:
        trainer = SpecialistTrainer(key, epochs, lr, accum, drop, rank_w)
        trainer.train()
    
    logging.info("\n✅ TREINAMENTO SEMANAL CONCLUÍDO PARA TODOS OS ESPECIALISTAS.")

if __name__ == "__main__":
    main()

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

# Configurações Globais (Treino Natural: CPU para estabilidade)
DEVICE = torch.device('cpu') 
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
    def __init__(self, region_key, epochs=60, lr=0.05, batch_size=16, dropout=0.3, ranking_weight=20.0):
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
        logging.info(f"🚀 INICIANDO TREINO NATURAL: {region_label}")
        logging.info(f"📊 Config: LR={self.lr}, Batch={self.batch_size}, Épocas={self.epochs}, RankWeight={self.ranking_weight}")
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
        
        adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32)
        adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32)
        adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)

        features_norm = features.copy()
        for c in range(C):
            mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
            features_norm[:, :, c] = (features[:, :, c] - mean) / std

        X_all, y_all, weights_all = [], [], []
        df_temp = pd.DataFrame({'date': dates, 'crimes': features[:, :, 0].sum(axis=0)})
        avg_crimes = df_temp['crimes'].mean() + 1e-6
        month_weights = {m: max(0.8, min(1.3, val/avg_crimes)) for m, val in df_temp.groupby(df_temp['date'].dt.month)['crimes'].mean().items()}
        day_weights = {d: max(0.8, min(1.3, val/avg_crimes)) for d, val in df_temp.groupby(df_temp['date'].dt.dayofweek)['crimes'].mean().items()}

        logging.info(f"📦 Vetorizando janelas temporais para {region_label}...")
        for t in range(WINDOW, T_total - PREDICT_HORIZON):
            x_tensor = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1)
            y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
            y_target = y_raw + (0.2 * torch.matmul(adj_dense, y_raw))
            if y_target.max() > 0: y_target = y_target / y_target.max()
            t_mult = month_weights.get(dates[t].month, 1.0) * day_weights.get(dates[t].dayofweek, 1.0)
            X_all.append(x_tensor)
            y_all.append(y_target)
            weights_all.append(t_mult)

        X_all, y_all = torch.stack(X_all), torch.stack(y_all)
        weights_all = torch.tensor(weights_all, dtype=torch.float32)

        lastro_days = 60
        total_idx = len(X_all)
        train_limit = total_idx - lastro_days - GAP
        X_train, y_train, w_train = X_all[:train_limit], y_all[:train_limit], weights_all[:train_limit]
        X_val, y_val = X_all[total_idx-lastro_days:], y_all[total_idx-lastro_days:]
        
        train_ds = TensorDataset(X_train, y_train, w_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=self.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # OneCycleLR Scheduler para convergência agressiva e rápida
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.lr, 
            steps_per_epoch=steps_per_epoch, 
            epochs=self.epochs
        )

        logging.info(f"🎬 Iniciando Treino: {len(X_train)} amostras | Device: {DEVICE}")

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            start_epoch = time.time()
            for i, (bx, by, bw) in enumerate(train_loader):
                batch_start = time.time()
                optimizer.zero_grad()
                pred = model(bx, [adj_geo, adj_conf]).squeeze(-1)
                
                loss_reg = (F.smooth_l1_loss(pred, by, reduction='none').mean(dim=1) * bw).mean()
                loss_rank = torch.tensor(0.0)
                if self.ranking_weight > 0:
                    _, top_idx = torch.topk(by, 20, dim=1)
                    loss_rank = F.relu(0.5 - (pred.gather(1, top_idx) - pred.mean(dim=1, keepdim=True))).mean()
                
                loss = loss_reg + self.ranking_weight * loss_rank
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() # OneCycleLR atualiza a cada batch
                
                total_loss += loss.item()

                if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                    elapsed = time.time() - batch_start
                    
                    # Calcular P10 e P20 no Batch
                    batch_p10, batch_p20 = [], []
                    pred_np = pred.detach().cpu().numpy()
                    by_np = by.cpu().numpy()
                    for b_idx in range(len(by_np)):
                        if np.sum(by_np[b_idx]) > 0:
                            t_true = np.argsort(by_np[b_idx])[::-1]
                            t_pred = np.argsort(pred_np[b_idx])[::-1]
                            batch_p10.append(len(set(t_true[:10]) & set(t_pred[:10])) / 10.0)
                            batch_p20.append(len(set(t_true[:20]) & set(t_pred[:20])) / 20.0)
                    
                    avg_p10 = np.mean(batch_p10) if batch_p10 else 0.0
                    avg_p20 = np.mean(batch_p20) if batch_p20 else 0.0
                    current_lr = scheduler.get_last_lr()[0]
                    
                    logging.info(f"   > E{epoch+1:02d} B{i+1:03d}/{len(train_loader)} | LR: {current_lr:.4f} | Loss: {loss.item():.4f} | P@10: {avg_p10*100:.1f}% | P@20: {avg_p20*100:.1f}% | Time: {elapsed:.2f}s")
            
            epoch_time = time.time() - start_epoch
            
            # Validar a cada 5 épocas ou na primeira
            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                p20_list = []
                with torch.no_grad():
                    val_pred = model(X_val, [adj_geo, adj_conf]).squeeze(-1)
                    for k in range(len(y_val)):
                        y_true_np, y_pred_np = y_val[k].numpy(), val_pred[k].numpy()
                        if np.sum(y_true_np) > 0:
                            t_true = np.argsort(y_true_np)[-20:]
                            t_pred = np.argsort(y_pred_np)[-20:]
                            p20_list.append(len(set(t_true) & set(t_pred)) / 20.0)
                
                current_p20 = np.mean(p20_list or [0])
                logging.info(f"✅ [{region_label}] E{epoch+1:02d} FINAL | Val P@20: {current_p20*100:.1f}% | Epoch Time: {epoch_time:.1f}s")
                
                if current_p20 > self.best_p20:
                    self.best_p20 = current_p20
                    save_path = os.path.join(ROOT_DIR, 'models', 'active', f'{self.region_key}_model.pth')
                    torch.save({'model_state_dict': model.state_dict(), 'p20': current_p20}, save_path)
                    logging.info(f"🏆 NOVO RECORDE {region_label}: P@20={current_p20*100:.1f}%")

def main():
    os.makedirs(os.path.join(ROOT_DIR, 'models', 'active'), exist_ok=True)
<<<<<<< HEAD
    
    # Configurações Elite ISM (Ajustadas para performance e convergência)
    tasks = [
        ('fortaleza', 60, 0.01, 16, 0.4, 20.0),
        ('rmf', 60, 0.01, 16, 0.4, 25.0),
        ('interior', 60, 0.01, 16, 0.3, 20.0)
=======
    # Configuração Natural (Ajustada contra overfitting, BATCH 32)
    tasks = [
        ('fortaleza', 60, 0.01, 32, 0.3, 15.0),
        ('rmf', 60, 0.01, 32, 0.3, 15.0),
        ('interior', 60, 0.01, 32, 0.3, 15.0)
>>>>>>> c45913460fadee72d09820642b87fdf125b7e792
    ]
    
    for key, epochs, lr, bs, drop, rank_w in tasks:
<<<<<<< HEAD
        try:
            trainer = SpecialistTrainer(key, epochs, lr, bs, drop, rank_w)
            trainer.train()
        except Exception as e:
            logging.error(f"❌ Erro ao treinar especialista {key.upper()}: {str(e)}")
            continue
    
    logging.info("\n✅ CICLO DE TREINAMENTO CONCLUÍDO.")
=======
        trainer = SpecialistTrainer(key, epochs, lr, bs, drop, rank_w)
        trainer.train()
    logging.info("\n✅ TREINO NATURAL CONCLUÍDO COM SUCESSO.")
>>>>>>> c45913460fadee72d09820642b87fdf125b7e792

if __name__ == "__main__":
    main()

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

# Adicionar raiz ao path para imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        logging.FileHandler("logs/training_fortaleza_breakthrough.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configurações Globais
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 120
PREDICT_HORIZON = 7
GAP = PREDICT_HORIZON + 7 # Safety Gap de 14 dias

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def calculate_priority_weights(features, dates):
    cvli_total_per_node = features[:, :, 0].sum(axis=1)
    spatial_weights = 1.0 + (cvli_total_per_node / (cvli_total_per_node.max() + 1e-6)) * 0.4
    df_temp = pd.DataFrame({'date': pd.to_datetime(dates), 'crimes': features[:, :, 0].sum(axis=0)})
    df_temp['month'] = df_temp['date'].dt.month
    df_temp['dow'] = df_temp['date'].dt.dayofweek
    avg_crimes = df_temp['crimes'].mean() + 1e-6
    month_avg = df_temp.groupby('month')['crimes'].mean()
    month_weights = {m: max(0.8, min(1.3, val/avg_crimes)) for m, val in month_avg.items()}
    dow_avg = df_temp.groupby('dow')['crimes'].mean()
    day_weights = {d: max(0.8, min(1.3, val/avg_crimes)) for d, val in dow_avg.items()}
    return spatial_weights, month_weights, day_weights

class SpecialistTrainer:
    def __init__(self, region_key, epochs=80, lr=0.005, grad_accum=16, dropout=0.4, ranking_weight=30.0):
        self.region_key = region_key
        self.epochs = epochs
        self.lr = lr
        self.grad_accum = grad_accum
        self.dropout = dropout
        self.ranking_weight = ranking_weight
        self.best_p20 = 0.0
        
    def train(self):
        region_label = self.region_key.upper()
        logging.info("\n" + "="*80)
        logging.info(f"🚀 INICIANDO TREINAMENTO ESTABILIZADO: {region_label}")
        logging.info("="*80)
        
        data_path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        features = data['node_features'] 
        dates = pd.to_datetime(data['dates'])
        N, T_total, C = features.shape
        
        spatial_weights_np, month_weights_map, day_weights_map = calculate_priority_weights(features, dates)
        spatial_weights = torch.tensor(spatial_weights_np, dtype=torch.float32).to(DEVICE)
        adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
        adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
        adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)

        features_norm = features.copy()
        for c in range(C):
            mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
            features_norm[:, :, c] = (features[:, :, c] - mean) / std

        X_list, y_list, info_list = [], [], []
        for t in range(WINDOW, T_total - PREDICT_HORIZON):
            x_tensor = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
            # Suavização de Label: Mistura com vizinhos para inércia
            y_target = y_raw + (0.2 * torch.matmul(adj_dense, y_raw))
            if y_target.max() > 0: y_target = y_target / y_target.max()
            current_date = dates[t]
            info_list.append({'month': current_date.month, 'dow': current_date.dayofweek, 'total_crimes': y_raw.sum().item()})
            X_list.append(x_tensor)
            y_list.append(y_target.unsqueeze(0))
            
        lastro_days, val_days = 90, 60
        total_idx = len(X_list)
        available_limit = total_idx - lastro_days - GAP
        available_idx = list(range(available_limit))
        val_idx = random.sample(available_idx, val_days)
        train_idx_base = [i for i in available_idx if i not in val_idx]
        lastro_idx = list(range(total_idx - lastro_days, total_idx))
        
        # OVERSAMPLING: Repetir dias quentes (acima da mediana de crimes)
        train_crime_median = np.median([info_list[i]['total_crimes'] for i in train_idx_base])
        high_crime_idx = [i for i in train_idx_base if info_list[i]['total_crimes'] > train_crime_median]
        train_indices_final = train_idx_base + high_crime_idx + high_crime_idx
        
        train_X = [X_list[i] for i in train_idx_base] # Para o dataset
        train_y = [y_list[i] for i in train_idx_base]
        train_info = [info_list[i] for i in train_idx_base]
        lastro_X = [X_list[i] for i in lastro_idx]
        lastro_y = [y_list[i] for i in lastro_idx]
        
        model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=self.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-3)
        
        def criterion(pred, target, info):
            pred, target = pred.squeeze(), target.squeeze()
            t_mult = month_weights_map.get(info['month'], 1.0) * day_weights_map.get(info['dow'], 1.0)
            k = 25 
            top_val, top_idx = torch.topk(target, min(k, len(target)))
            w = spatial_weights.clone()
            w[top_idx] = w[top_idx] * 6.0 * (1.0 + target[top_idx])
            loss_reg = (w * F.smooth_l1_loss(pred, target, reduction='none')).mean()
            if top_val.sum() == 0: return loss_reg * t_mult
            num_neg = 50
            neg_idx = torch.randint(0, len(target), (num_neg,), device=target.device)
            p_h, p_l = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
            t_h, t_l = target[top_idx].unsqueeze(1), target[neg_idx].unsqueeze(0)
            margin = 0.3 + (F.relu(t_h - t_l) * 0.5)
            loss_rank = (F.relu(margin - (p_h - p_l)) * (t_h > t_l).float()).sum() / (num_neg * k)
            return (loss_reg + self.ranking_weight * loss_rank) * t_mult

        steps_per_epoch = (len(train_indices_final) // self.grad_accum) + 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=steps_per_epoch, epochs=self.epochs)
        
        logging.info(f"🎬 Iniciando Treinamento: {self.epochs} épocas | Oversampling Ativo")
        total_batches = len(train_indices_final) // self.grad_accum

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            indices_shuffled = train_indices_final.copy()
            random.shuffle(indices_shuffled)
            steps = 0
            for idx in indices_shuffled:
                bx, by, bi = X_list[idx].to(DEVICE), y_list[idx].to(DEVICE), info_list[idx]
                pred = model(bx, [adj_geo, adj_conf])
                loss_obj = criterion(pred, by, bi)
                loss = loss_obj / self.grad_accum
                loss.backward()
                steps += 1
                if steps % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    current_batch = steps // self.grad_accum
                    if current_batch % 20 == 0 or current_batch == 1:
                        with torch.no_grad():
                            y_true, y_pred = by.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy()
                            p10, p20 = 0.0, 0.0
                            if np.sum(y_true) > 0:
                                t_true, t_pred = np.argsort(y_true)[::-1], np.argsort(y_pred)[::-1]
                                p10 = len(set(t_true[:10]) & set(t_pred[:10])) / 10.0
                                p20 = len(set(t_true[:20]) & set(t_pred[:20])) / 20.0
                        logging.info(f"   E{epoch+1:02d} | Batch [{current_batch:03d}/{total_batches:03d}] | Loss: {loss_obj.item():.4f} | P@10: {p10*100:>5.1f}% | P@20: {p20*100:>5.1f}%")

            model.eval()
            p10_l, p20_l = [], []
            with torch.no_grad():
                for i in range(len(lastro_X)):
                    lx, ly = lastro_X[i].to(DEVICE), lastro_y[i].to(DEVICE)
                    lpred = model(lx, [adj_geo, adj_conf])
                    ly_np, lp_np = ly.squeeze().cpu().numpy(), lpred.squeeze().cpu().numpy()
                    if np.sum(ly_np) > 0:
                        t_true, t_pred = np.argsort(ly_np)[::-1], np.argsort(lp_np)[::-1]
                        p10_l.append(len(set(t_true[:10]) & set(t_pred[:10])) / 10.0)
                        p20_l.append(len(set(t_true[:20]) & set(t_pred[:20])) / 20.0)
            real_p10, real_p20 = np.mean(p10_l or [0]), np.mean(p20_l or [0])
            logging.info(f"[{region_label}] E{epoch+1:02d} | REALITY P@10: {real_p10*100:.1f}% | P@20: {real_p20*100:.1f}%")
            if real_p20 > self.best_p20:
                self.best_p20 = real_p20
                save_path = os.path.join(ROOT_DIR, 'models', 'active', f'{self.region_key}_model_new.pth')
                torch.save({'model_state_dict': model.state_dict(), 'p20': real_p20}, save_path)
                logging.info(f"🏆 NOVO RECORDE {region_label}: P@20={real_p20*100:.1f}%")

def main():
    os.makedirs(os.path.join(ROOT_DIR, 'models', 'active'), exist_ok=True)
    tasks = [('fortaleza', 80, 0.005, 16, 0.4, 30.0)]
    for key, epochs, lr, accum, drop, rank_w in tasks:
        trainer = SpecialistTrainer(key, epochs, lr, accum, drop, rank_w)
        trainer.train()

if __name__ == "__main__":
    main()

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

# Adicionar raiz ao path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.core.architectures import DeepSTGAT_32 # Arquitetura mais leve para poucas cidades
except ImportError:
    sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))
    from architectures import DeepSTGAT_32

# Configuração de Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_rmf_elite.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

EPOCHS = 100
LR = 0.005 # Base LR para OneCycle
GRADIENT_ACCUMULATION_STEPS = 4 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_processed_data(region_key):
    path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{region_key}.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

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

def train_specialist(region_key, ModelClass):
    region_label = region_key.upper()
    logging.info(f"⚡ ESTRATÉGIA ELITE RMF: {region_label}")
    
    data = load_processed_data(region_key)
    features = data['node_features'] 
    dates = pd.to_datetime(data['dates'])
    
    spatial_weights_np, month_weights_map, day_weights_map = calculate_priority_weights(features, dates)
    spatial_weights = torch.tensor(spatial_weights_np, dtype=torch.float32).to(DEVICE)
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    
    # JANELA AMPLA PARA ELIMINAR RUÍDO (Tentativa 24)
    WINDOW, PREDICT_HORIZON = 120, 7
    N, T_total, C = features.shape
    
    features_norm = features.copy()
    for c in range(C):
        mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
        features_norm[:, :, c] = (features[:, :, c] - mean) / std

    X_list, y_list, info_list = [], [], []
    adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)

    for t in range(WINDOW, T_total - PREDICT_HORIZON):
        x_tensor = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
        y_target = y_raw + (0.3 * torch.matmul(adj_dense, y_raw))
        if y_target.max() > 0: y_target = y_target / y_target.max()
        current_date = dates[t]
        info_list.append({'month': current_date.month, 'dow': current_date.dayofweek})
        X_list.append(x_tensor)
        y_list.append(y_target.unsqueeze(0))
        
    lastro_days, val_days, gap = 90, 60, PREDICT_HORIZON + 7
    total_idx = len(X_list)
    available_limit = total_idx - lastro_days - gap
    available_idx = list(range(available_limit))
    val_idx = random.sample(available_idx, val_days)
    train_idx_base = [i for i in available_idx if i not in val_idx]
    lastro_idx = list(range(total_idx - lastro_days, total_idx))
    
    train_X = [X_list[i] for i in train_idx_base]
    train_y = [y_list[i] for i in train_idx_base]
    train_info = [info_list[i] for i in train_idx_base]
    val_X = [X_list[i] for i in val_idx]
    val_y = [y_list[i] for i in val_idx]
    val_info = [info_list[i] for i in val_idx]
    lastro_X = [X_list[i] for i in lastro_idx]
    lastro_y = [y_list[i] for i in lastro_idx]
    lastro_info = [info_list[i] for i in lastro_idx]
    
    # DROPOUT 0.3 para RMF (menos que Fortaleza por ter menos nós)
    model = ModelClass(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    def criterion(pred, target, info):
        pred, target = pred.squeeze(), target.squeeze()
        t_mult = month_weights_map.get(info['month'], 1.0) * day_weights_map.get(info['dow'], 1.0)
        
        # FOCO EM P@5 (Top 5 de 18 cidades)
        k = 5 
        top_val, top_idx = torch.topk(target, min(k, len(target)))
        
        w = spatial_weights.clone()
        w[top_idx] = w[top_idx] * 6.0 * (1.0 + target[top_idx]) 
        
        loss_reg = (w * F.smooth_l1_loss(pred, target, reduction='none')).mean()
        
        if top_val.sum() == 0: return loss_reg * t_mult
        
        num_neg = 12 
        neg_idx = torch.randint(0, len(target), (num_neg,), device=target.device)
        
        p_h, p_l = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
        t_h, t_l = target[top_idx].unsqueeze(1), target[neg_idx].unsqueeze(0)
        
        margin = 0.35 + (F.relu(t_h - t_l) * 0.7)
        loss_rank = (F.relu(margin - (p_h - p_l)) * (t_h > t_l).float()).sum() / (num_neg * k)
        
        # RANKING 25.0 para consolidar Top-5 e Top-10
        return (loss_reg + 25.0 * loss_rank) * t_mult

    steps_per_epoch = (len(train_X) // GRADIENT_ACCUMULATION_STEPS) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)
    
    total_steps = len(train_X) // GRADIENT_ACCUMULATION_STEPS

    logging.info(f"🎬 Iniciando Treino RMF ELITE: {EPOCHS} épocas | Foco P@5/P@10")
    logging.info(f"{'='*80}")
    logging.info(f"{'PASSO':<15} | {'LR':<10} | {'LOSS':<10} | {'P@5':<8} | {'P@10':<8}")
    logging.info(f"{'='*80}")

    best_combined = -1.0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        steps = 0
        for idx in indices:
            bx, by, bi = train_X[idx].to(DEVICE), train_y[idx].to(DEVICE), train_info[idx]
            pred = model(bx, [adj_geo, adj_conf])
            loss = criterion(pred, by, bi) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            steps += 1
            if steps % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                optimizer.zero_grad()
                current_step = steps // GRADIENT_ACCUMULATION_STEPS
                if current_step % 40 == 0:
                    y_true, y_pred = by.squeeze().cpu().detach().numpy(), pred.squeeze().cpu().detach().numpy()
                    p5, p10 = 0.0, 0.0
                    if np.sum(y_true) > 0:
                        t_true, t_pred = np.argsort(y_true)[::-1], np.argsort(y_pred)[::-1]
                        p5 = len(set(t_true[:5]) & set(t_pred[:5])) / 5.0
                        p10 = len(set(t_true[:10]) & set(t_pred[:10])) / 10.0
                    logging.info(f"E{epoch+1:02d} [{current_step:03d}/{total_steps:03d}] | {current_lr:.6f} | {loss.item()*GRADIENT_ACCUMULATION_STEPS:.6f} | {p5*100:>5.1f}% | {p10*100:>5.1f}%")
        
        model.eval()
        p5_l, p10_l = [], []
        with torch.no_grad():
            for i in range(len(lastro_X)):
                lx, ly, li = lastro_X[i].to(DEVICE), lastro_y[i].to(DEVICE), lastro_info[i]
                lpred = model(lx, [adj_geo, adj_conf])
                ly_np, lp_np = ly.squeeze().cpu().numpy(), lpred.squeeze().cpu().numpy()
                if np.sum(ly_np) > 0:
                    t_true, t_pred = np.argsort(ly_np)[::-1], np.argsort(lp_np)[::-1]
                    p5_l.append(len(set(t_true[:5]) & set(t_pred[:5])) / 5.0)
                    p10_l.append(len(set(t_true[:10]) & set(t_pred[:10])) / 10.0)
        
        real_p5 = np.mean(p5_l or [0])
        real_p10 = np.mean(p10_l or [0])
        combined_score = (real_p5 * 0.6) + (real_p10 * 0.4)
        
        logging.info(f"🏁 Epoch {epoch+1:03d} | REALITY P@5: {real_p5*100:.1f}% | P@10: {real_p10*100:.1f}%")
        
        if combined_score > best_combined:
            best_combined = combined_score
            save_path = os.path.join(ROOT_DIR, 'models', 'active', f'rmf_model_elite.pth')
            torch.save({
                'model_state_dict': model.state_dict(), 
                'p5_record': real_p5,
                'p10_record': real_p10,
                'combined_score': combined_score
            }, save_path)
            logging.info(f"🏆 NOVO RECORDE RMF (P@5): {real_p5*100:.1f}% | (P@10): {real_p10*100:.1f}%")
        
        if real_p5 >= 0.80 and real_p10 >= 0.85:
            logging.info(f"🎯 META ALCANÇADA: P@5={real_p5*100:.1f}%, P@10={real_p10*100:.1f}%")
            break

if __name__ == "__main__":
    from src.core.architectures import DeepSTGAT_64
    train_specialist('rmf', DeepSTGAT_64)

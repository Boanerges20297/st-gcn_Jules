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
from sklearn.model_selection import TimeSeriesSplit

# Adicionar raiz ao path para imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.core.architectures import DeepSTGAT_64
except ImportError:
    sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))
    from architectures import DeepSTGAT_64

# Configuração de Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/cv_fortaleza.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configurações de Treino
EPOCHS = 20 # Menos épocas por fold para agilidade (total = 5 folds * 20 = 100 épocas de esforço)
LR = 0.02 
GRADIENT_ACCUMULATION_STEPS = 24 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SPLITS = 5

def load_processed_data(region_key):
    path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{region_key}.pkl')
    logging.info(f"Carregando dados de: {path}")
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

def train_fold(fold_idx, train_idx, val_idx, X_list, y_list, info_list, adj_geo, adj_conf, spatial_weights, month_weights_map, day_weights_map, N, C, WINDOW):
    logging.info(f"\n>>> INICIANDO FOLD {fold_idx+1}/{N_SPLITS}")
    logging.info(f"    Treino: {len(train_idx)} amostras | Teste: {len(val_idx)} amostras")
    
    train_X = [X_list[i] for i in train_idx]
    train_y = [y_list[i] for i in train_idx]
    train_info = [info_list[i] for i in train_idx]
    
    val_X = [X_list[i] for i in val_idx]
    val_y = [y_list[i] for i in val_idx]
    val_info = [info_list[i] for i in val_idx]
    
    model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    steps_per_epoch = (len(train_X) * 2 // GRADIENT_ACCUMULATION_STEPS) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR*3.0, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)
    
    def criterion(pred, target, info):
        pred, target = pred.squeeze(), target.squeeze()
        t_mult = month_weights_map.get(info['month'], 1.0) * day_weights_map.get(info['dow'], 1.0)
        k = 30
        top_val, top_idx = torch.topk(target, min(k, len(target)))
        w = spatial_weights.clone()
        w[top_idx] = w[top_idx] * 4.0 * (1.0 + target[top_idx])
        loss_reg = (w * F.smooth_l1_loss(pred, target, reduction='none')).mean()
        if top_val.sum() == 0: return loss_reg * t_mult
        num_neg = 50
        neg_idx = torch.randint(0, len(target), (num_neg,), device=target.device)
        p_h, p_l = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
        t_h, t_l = target[top_idx].unsqueeze(1), target[neg_idx].unsqueeze(0)
        margin = 0.2 + (F.relu(t_h - t_l) * 0.5)
        loss_rank = (F.relu(margin - (p_h - p_l)) * (t_h > t_l).float()).sum() / (num_neg * k)
        return (loss_reg + 0.3 * loss_rank) * t_mult

    best_p20 = 0.0
    
    # Oversampling para o fold atual
    day_sev = [torch.sum(y).item() for y in train_y]
    high_idx = [i for i, s in enumerate(day_sev) if s > np.median(day_sev)]
    train_indices_base = list(range(len(train_X))) + high_idx + high_idx

    # Calcular limiares de intensidade (tercis)
    all_y_sums = [torch.sum(y).item() for y in train_y]
    threshold_low = np.percentile(all_y_sums, 33)
    threshold_high = np.percentile(all_y_sums, 66)
    
    logging.info(f"    Limiares de Intensidade: Calmo < {threshold_low:.2f} | Morno {threshold_low:.2f}-{threshold_high:.2f} | Quente > {threshold_high:.2f}")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        np.random.shuffle(train_indices_base)
        
        loss_calm, count_calm = 0.0, 0
        loss_warm, count_warm = 0.0, 0
        loss_hot, count_hot = 0.0, 0
        
        # Batch loop simplificado
        for step, idx in enumerate(train_indices_base):
            bx, by, bi = train_X[idx].to(DEVICE), train_y[idx].to(DEVICE), train_info[idx]
            pred = model(bx, [adj_geo, adj_conf])
            loss = criterion(pred, by, bi) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Classificar intensidade do dia
            daily_sum = by.sum().item()
            loss_val = loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            if daily_sum < threshold_low:
                loss_calm += loss_val
                count_calm += 1
            elif daily_sum < threshold_high:
                loss_warm += loss_val
                count_warm += 1
            else:
                loss_hot += loss_val
                count_hot += 1
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_loss_calm = loss_calm / max(1, count_calm)
        avg_loss_warm = loss_warm / max(1, count_warm)
        avg_loss_hot = loss_hot / max(1, count_hot)
        
        # Validação
        model.eval()
        p20_l = []
        with torch.no_grad():
            for i in range(len(val_X)):
                vx, vy, vi = val_X[i].to(DEVICE), val_y[i].to(DEVICE), val_info[i]
                vpred = model(vx, [adj_geo, adj_conf])
                vy_np, vp_np = vy.squeeze().cpu().numpy(), vpred.squeeze().cpu().numpy()
                if np.sum(vy_np) > 0:
                    t_true, t_pred = np.argsort(vy_np)[::-1], np.argsort(vp_np)[::-1]
                    p20_l.append(len(set(t_true[:20]) & set(t_pred[:20])) / 20.0)
        
        avg_p20 = np.mean(p20_l or [0])
        if avg_p20 > best_p20:
            best_p20 = avg_p20
            # Salvar melhor do fold
            torch.save(model.state_dict(), f'models/test/fold_{fold_idx+1}_best.pth')
            
        if (epoch+1) % 1 == 0:
            logging.info(f"    [Fold {fold_idx+1}] Epoch {epoch+1}/{EPOCHS} | Loss: C={avg_loss_calm:.4f} M={avg_loss_warm:.4f} Q={avg_loss_hot:.4f} | P@20 Valid: {avg_p20*100:.1f}% (Best: {best_p20*100:.1f}%)")
            
    logging.info(f"✅ FOLD {fold_idx+1} CONCLUÍDO. Melhor P@20: {best_p20*100:.1f}%")
    return best_p20

def run_cv():
    logging.info(f"="*50)
    logging.info(f"🧪 INICIANDO CROSS-VALIDATION (5-FOLD TimeSeriesSplit)")
    logging.info(f"="*50)
    
    data = load_processed_data('fortaleza')
    features = data['node_features'] 
    dates = pd.to_datetime(data['dates'])
    
    spatial_weights_np, month_weights_map, day_weights_map = calculate_priority_weights(features, dates)
    spatial_weights = torch.tensor(spatial_weights_np, dtype=torch.float32).to(DEVICE)
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    
    WINDOW, PREDICT_HORIZON = 30, 7
    N, T_total, C = features.shape
    features_norm = features.copy()
    for c in range(C):
        mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
        features_norm[:, :, c] = (features[:, :, c] - mean) / std

    X_list, y_list, info_list = [], [], []
    adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)
    dates = pd.to_datetime(data['dates'])

    for t in range(WINDOW, T_total - PREDICT_HORIZON):
        x_tensor = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
        y_target = y_raw + (0.3 * torch.matmul(adj_dense, y_raw))
        if y_target.max() > 0: y_target = y_target / y_target.max()
        current_date = dates[t]
        info_list.append({'month': current_date.month, 'dow': current_date.dayofweek})
        X_list.append(x_tensor)
        y_list.append(y_target.unsqueeze(0))

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_scores = []
    
    # Preparar índices gerais
    all_indices = np.arange(len(X_list))
    
    for fold_idx, (train_index, test_index) in enumerate(tscv.split(all_indices)):
        # Garantir gap de segurança entre treino e teste no fold
        gap = PREDICT_HORIZON + 7
        if len(test_index) > gap:
            real_test_index = test_index[gap:] # Descarta o início do teste para garantir gap
        else:
            real_test_index = test_index # Se for muito curto, usa o que tem (risco de leakage mínimo em CV rápido)
            
        score = train_fold(fold_idx, train_index, real_test_index, X_list, y_list, info_list, 
                           adj_geo, adj_conf, spatial_weights, month_weights_map, day_weights_map, N, C, WINDOW)
        fold_scores.append(score)

    avg_score = np.mean(fold_scores)
    logging.info(f"="*50)
    logging.info(f"📊 RESULTADO FINAL DO CROSS-VALIDATION")
    logging.info(f"Scores por Fold: {[f'{s*100:.1f}%' for s in fold_scores]}")
    logging.info(f"Média P@20: {avg_score*100:.1f}%")
    logging.info(f"="*50)

if __name__ == "__main__":
    run_cv()

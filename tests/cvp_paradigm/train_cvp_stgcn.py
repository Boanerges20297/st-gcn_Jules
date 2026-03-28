import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import random
import time
from datetime import datetime

# Adicionar raiz ao path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

from tests.cvp_paradigm.stgcn_architecture import DeepSTGCN_CVP

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PKL_PATH = 'tests/cvp_paradigm/processed_fortaleza_CVP.pkl'
LOG_PATH = 'logs/training_CVP_STGCN.log'
MODEL_SAVE = 'tests/cvp_paradigm/cvp_stgcn_model.pth'

# --- HIPERPARAMETROS ATUALIZADOS ---
EPOCHS = 100
LR = 0.02
WINDOW = 14
HORIZON = 7
DROPOUT = 0.5
RANK_WEIGHT = 20.0

# Logger Detalhado (UTF-8)
file_handler = logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[file_handler, stream_handler])

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = np.diag(d_inv_sqrt)
    return d_mat.dot(adj).dot(d_mat)

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def calculate_precision(pred, target, k_list=[10, 20]):
    res = {}
    pred, target = pred.flatten(), target.flatten()
    idx_p = np.argsort(pred)[::-1]
    idx_t = np.argsort(target)[::-1]
    for k in k_list:
        res[f'P@{k}'] = len(set(idx_p[:k]) & set(idx_t[:k])) / k
    return res

def train():
    logging.info(f"--- TREINO STGCN v5.1: TELEMETRIA DETALHADA (DEVICE: {DEVICE}) ---")
    
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    features = data['node_features']
    adj_norm = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    
    N, T_total, C = features.shape
    
    features_norm = features.copy()
    for c in range(C):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features[:, :, c] - m) / s

    X_list, y_list = [], []
    for t in range(WINDOW, T_total - HORIZON):
        x = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1)
        y_target = torch.tensor(features[:, t:t+HORIZON, 0].sum(axis=1), dtype=torch.float32)
        if y_target.max() > 0: y_target /= y_target.max()
        X_list.append(x.unsqueeze(0))
        y_list.append(y_target.unsqueeze(0))

    split = int(len(X_list) * 0.8)
    train_X, train_y = X_list[:split], y_list[:split]
    val_X, val_y = X_list[split:], y_list[split:]

    model = DeepSTGCN_CVP(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=DROPOUT).to(DEVICE)
    model.set_adj(adj_norm)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_X), epochs=EPOCHS)
    
    ranking_criterion = nn.MarginRankingLoss(margin=0.15)

    best_p10 = 0
    for epoch in range(EPOCHS):
        model.train()
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        
        for step, i in enumerate(indices):
            bx, by = train_X[i].to(DEVICE), train_y[i].to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            
            mse = F.mse_loss(pred.squeeze(), by.squeeze())
            k = 15
            _, top_idx = torch.topk(by.squeeze(), k)
            _, mid_idx = torch.topk(by.squeeze(), k + 20)
            mid_idx = mid_idx[20:]
            y_rank = torch.ones(k).to(DEVICE)
            rank_loss = ranking_criterion(pred.squeeze()[top_idx], pred.squeeze()[mid_idx], y_rank)
            
            loss = mse + RANK_WEIGHT * rank_loss
            loss.backward()
            
            g_norm = get_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            if step % 100 == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                p_metrics = calculate_precision(pred.detach().cpu().numpy(), by.cpu().numpy())
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"[{ts}] EP {epoch+1:03d} | Step {step:04d}/{len(train_X)} | Loss: {loss.item():.4f} | Grad: {g_norm:.3f} | P@10: {p_metrics['P@10']*100:.1f}% | P@20: {p_metrics['P@20']*100:.1f}% | LR: {current_lr:.6f}")
        
        model.eval()
        v_p10, v_p20, v_loss = [], [], 0
        with torch.no_grad():
            for i in range(len(val_X)):
                vx, vy = val_X[i].to(DEVICE), val_y[i].to(DEVICE)
                vp = model(vx).squeeze().cpu().numpy()
                vt = vy.squeeze().cpu().numpy()
                v_loss += F.mse_loss(torch.tensor(vp), vy.squeeze()).item()
                if vt.sum() > 0:
                    p_res = calculate_precision(vp, vt)
                    v_p10.append(p_res['P@10'])
                    v_p20.append(p_res['P@20'])
        
        avg_p10 = np.mean(v_p10) if v_p10 else 0
        avg_p20 = np.mean(v_p20) if v_p20 else 0
        logging.info("=" * 110)
        logging.info(f"FIM EP {epoch+1:03d} | Val Loss: {v_loss/len(val_X):.4f} | P@10 FINAL: {avg_p10*100:.2f}% | P@20 FINAL: {avg_p20*100:.2f}%")
        logging.info("=" * 110)

        if avg_p10 > best_p10:
            best_p10 = avg_p10
            torch.save(model.state_dict(), MODEL_SAVE)
            logging.info(f"🏆 NEW RECORD: P@10 {best_p10*100:.2f}% (Saved)")

if __name__ == "__main__":
    train()

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import pandas as pd
import random
import time
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)
from src.core.architectures import DeepSTGAT_64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PKL_PATH = 'tests/cvp_paradigm/processed_fortaleza_CVP.pkl'
LOG_PATH = 'logs/training_CVP_PARADIGM.log'
MODEL_SAVE = 'tests/cvp_paradigm/cvp_paradigm_model.pth'

# --- HIPERPARÂMETROS OTIMIZADOS ---
EPOCHS = 60
LR_MAX = 0.004 # Reduzido para estabilidade
DROPOUT = 0.45 # Aumentado para regularização
RANKING_WEIGHT = 4.0
SUPRESSION_WEIGHT = 2.0

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

def calculate_precision(pred, target, k_list=[10, 20]):
    res = {}
    pred, target = pred.flatten(), target.flatten()
    idx_p = np.argsort(pred)[::-1]
    idx_t = np.argsort(target)[::-1]
    for k in k_list:
        res[f'P@{k}'] = len(set(idx_p[:k]) & set(idx_t[:k])) / k
    return res

def train():
    logging.info(f"--- TREINO CVP v3.0: OTIMIZADO (LR={LR_MAX}, DO={DROPOUT}, RW={RANKING_WEIGHT}) ---")
    
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    features = data['node_features']
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    
    N, T_total, C = features.shape
    WINDOW, HORIZON = 60, 7 
    
    features_norm = features.copy()
    for c in range(C):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features[:, :, c] - m) / s

    X_list, y_list = [], []
    for t in range(WINDOW, T_total - HORIZON):
        x = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y_target = torch.tensor(features[:, t:t+HORIZON, 0].sum(axis=1), dtype=torch.float32)
        if y_target.max() > 0: y_target /= y_target.max()
        X_list.append(x)
        y_list.append(y_target.unsqueeze(0))

    split = int(len(X_list) * 0.8)
    train_X, train_y = X_list[:split], y_list[:split]
    val_X, val_y = X_list[split:], y_list[split:]

    # Injetando Dropout customizado
    model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_X), epochs=EPOCHS)
    
    ranking_criterion = nn.MarginRankingLoss(margin=0.3) # Margem aumentada para 0.3

    def compute_loss(pred, target, x_input):
        pred, target = pred.squeeze(), target.squeeze()
        mse = F.mse_loss(pred, target)
        
        # Supressão (Peso 2.0)
        dominio_local = x_input[0, 2, :, -1] 
        supressao_loss = torch.mean(F.relu(pred - 0.2) * (dominio_local > 0.7).float())
        
        # Ranking (Peso 4.0) com Hard Negative Mining
        k = 15
        top_val, top_idx = torch.topk(target, k)
        # Pegamos o Mid-tier (bairros com crime médio) para comparar com os Top
        mid_val, mid_idx = torch.topk(target, k + 20)
        mid_idx = mid_idx[20:] # Pega do 20 ao 35
        
        y_rank = torch.ones(k).to(DEVICE)
        rank_loss = ranking_criterion(pred[top_idx], pred[mid_idx], y_rank)
        
        return mse + RANKING_WEIGHT * rank_loss + SUPRESSION_WEIGHT * supressao_loss

    best_p10 = 0
    for epoch in range(EPOCHS):
        model.train()
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        
        for step, i in enumerate(indices):
            bx, by = train_X[i].to(DEVICE), train_y[i].to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx, [adj_geo, adj_geo])
            loss = compute_loss(pred, by, bx)
            
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3) # Clip mais restrito
            optimizer.step()
            scheduler.step()
            
            if step % 100 == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                p_metrics = calculate_precision(pred.detach().cpu().numpy(), by.cpu().numpy())
                logging.info(f"[{ts}] EP {epoch+1:02d} | Step {step:04d} | Loss: {loss.item():.4f} | P@10: {p_metrics['P@10']*100:.1f}% | P@20: {p_metrics['P@20']*100:.1f}%")
        
        model.eval()
        v_p10, v_p20 = [], []
        with torch.no_grad():
            for i in range(len(val_X)):
                vx, vy = val_X[i].to(DEVICE), val_y[i].to(DEVICE)
                vp = model(vx, [adj_geo, adj_geo]).squeeze().cpu().numpy()
                vt = vy.squeeze().cpu().numpy()
                if vt.sum() > 0:
                    p_res = calculate_precision(vp, vt)
                    v_p10.append(p_res['P@10'])
                    v_p20.append(p_res['P@20'])
        
        avg_p10 = np.mean(v_p10) if v_p10 else 0
        avg_p20 = np.mean(v_p20) if v_p20 else 0
        logging.info("=" * 100)
        logging.info(f"FINAL EP {epoch+1:02d} | VAL P@10: {avg_p10*100:.2f}% | VAL P@20: {avg_p20*100:.2f}%")
        logging.info("=" * 100)
        logging.info(f"EP {epoch+1:02d} | LOSS: 0.0000 | P@10: {avg_p10*100:.2f}%") 

        if avg_p10 > best_p10:
            best_p10 = avg_p10
            torch.save(model.state_dict(), MODEL_SAVE)
            logging.info(f"🏆 RECORD: P@10 {best_p10*100:.2f}% (Saved)")

if __name__ == "__main__":
    train()

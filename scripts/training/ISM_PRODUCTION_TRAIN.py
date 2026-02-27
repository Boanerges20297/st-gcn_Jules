import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random
import logging
import sys

# Adicionar raiz ao path
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from src.core.architectures import DeepSTGAT_64
from src.core.data_processing import process_ism_data, normalize_adj

# --- ISM PRODUCTION CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/ISM_PRODUCTION_TRAIN.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 120
LR = 0.05
DROPOUT = 0.5
RANKING_WEIGHT = 20.0
EPOCHS = 60
GRAD_ACCUM = 32 

def train_specialist(region_key):
    logging.info(f"\n🚀 INICIANDO TREINO ISM: {region_key.upper()}")
    path = f'data/processed/processed_{region_key}.pkl'
    if not os.path.exists(path):
        logging.error(f"❌ Dataset não encontrado: {path}")
        return

    with open(path, 'rb') as f: data = pickle.load(f)
    
    nf, adj_geo_np, adj_conf_np = data['node_features'], data['adj_geo'], data['adj_conflict']
    adj_geo = torch.tensor(normalize_adj(adj_geo_np), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(adj_conf_np, dtype=torch.float32).to(DEVICE)
    N, T, C = nf.shape
    
    # Normalização Z-Score
    f_norm = nf.copy()
    for c in range(C):
        m, s = nf[:,:,c].mean(), nf[:,:,c].std() + 1e-5
        f_norm[:,:,c] = (nf[:,:,c] - m) / s

    X_list, y_list = [], []
    for t in range(WINDOW, T - 7):
        X_list.append(torch.tensor(f_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2,0,1).unsqueeze(0))
        y_list.append(torch.tensor(nf[:, t:t+7, 0].sum(axis=1), dtype=torch.float32).unsqueeze(0))
    
    val_size = 60
    train_X, train_y = X_list[:-val_size], y_list[:-val_size]
    val_X, val_y = X_list[-val_size:], y_list[-val_size:]
    
    model = DeepSTGAT_64(num_nodes=N, in_channels=29, time_steps=WINDOW, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = (len(train_X) // GRAD_ACCUM) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=total_steps, epochs=EPOCHS)

    best_val = 0
    for epoch in range(EPOCHS):
        model.train()
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        optimizer.zero_grad()
        
        for i, idx in enumerate(indices):
            bx, by = train_X[idx].to(DEVICE), train_y[idx].to(DEVICE)
            pred = model(bx, [adj_geo, adj_conf]).squeeze()
            target = by.squeeze()
            
            mse = F.smooth_l1_loss(pred, target / (target.max() + 1e-5))
            k_rank = 15 if region_key == 'fortaleza' else 10
            _, top_idx = torch.topk(target, min(k_rank, N))
            num_neg = min(30, N)
            neg_idx = torch.randint(0, N, (num_neg,), device=DEVICE)
            p_h, p_l = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
            rank_loss = F.relu(0.4 - (p_h - p_l)).mean()
            
            loss = (mse + RANKING_WEIGHT * rank_loss) / GRAD_ACCUM
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Log opcional de step pode ser adicionado aqui
        
        # Validação Final da Época
        model.eval()
        p10_acc, p20_acc = [], []
        with torch.no_grad():
            for i in range(len(val_X)):
                vx, vy = val_X[i].to(DEVICE), val_y[i].to(DEVICE)
                vpred = model(vx, [adj_geo, adj_conf]).squeeze().cpu().numpy()
                vtrue = vy.squeeze().cpu().numpy()
                if vtrue.sum() > 0:
                    p_idx10 = np.argsort(vpred)[::-1][:10]
                    t_idx10 = np.argsort(vtrue)[::-1][:10]
                    p10_acc.append(len(set(p_idx10) & set(t_idx10)) / 10.0)
                    
                    p_idx20 = np.argsort(vpred)[::-1][:20]
                    t_idx20 = np.argsort(vtrue)[::-1][:20]
                    p20_acc.append(len(set(p_idx20) & set(t_idx20)) / 20.0)
        
        mp10, mp20 = np.mean(p10_acc or [0]), np.mean(p20_acc or [0])
        logging.info(f"[{region_key.upper()}] Epoch {epoch+1:02d} | Val P@10: {mp10*100:.1f}% | Val P@20: {mp20*100:.1f}%")
        
        # Lógica de Recorde por Região (Jules Criteria)
        is_record = False
        if region_key in ['fortaleza', 'interior'] and mp20 > best_val:
            best_val = mp20
            is_record = True
        elif region_key == 'rmf' and mp10 > best_val:
            best_val = mp10
            is_record = True
            
        if is_record:
            torch.save({'model_state_dict': model.state_dict(), 'p10': mp10, 'p20': mp20}, f'models/active/{region_key}_model.pth')
            logging.info(f"🏆 NOVO RECORDE {region_key.upper()}: {best_val*100:.1f}%")

if __name__ == "__main__":
    # 1. Primeiro atualiza os nós e tensores com a lógica ISM
    process_ism_data()
    # 2. Treina os especialistas em sequência
    for reg in ['fortaleza', 'rmf', 'interior']:
        train_specialist(reg)
    logging.info("\n✅ CICLO ISM FINALIZADO COM SUCESSO.")

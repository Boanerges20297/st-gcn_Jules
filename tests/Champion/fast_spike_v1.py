import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
import logging

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from src.core.architectures import ShallowGAT
from scripts.training.Active.train_all_specialists import BinaryFocalRankingLoss

# Config Turbo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGION = 'fortaleza'
WINDOW = 120
LR = 0.001
EPOCHS = 3
BATCH_SIZE = 16  # Turbo: Processa 16 janelas por vez
SUBSET_SIZE = 400 # Turbo: Foca nas últimas 400 janelas (recentes)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(adj)

def run_fast_spike_v2():
    logging.info(f"🚀 Iniciando FAST SPIKE V2 (Turbo) | LR: {LR} | Heads: 16 | Batch: {BATCH_SIZE}")
    logging.info(f"💻 Rodando em: {DEVICE}")
    
    # Carregar dados
    pkl_path = os.path.join(ROOT_DIR, "data", "processed", f"processed_{REGION}.pkl")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    nf = data['node_features']
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    N, T, C = nf.shape
    C_ext = C + 2
    
    # Construção de Janelas (Subset Recente)
    X_list, Y_list = [], []
    all_t = range(T - WINDOW - 14)
    target_t = list(all_t)[-SUBSET_SIZE:] # Pega as últimas SUBSET_SIZE janelas
    
    logging.info(f"📐 Preparando {len(target_t)} janelas táticas...")
    for t in target_t:
        X_list.append(torch.tensor(nf[:, t:t+WINDOW, :].transpose(2, 0, 1), dtype=torch.float32))
        Y_list.append(torch.tensor(nf[:, t+WINDOW+14, 0], dtype=torch.float32))
    
    train_X = torch.stack(X_list).to(DEVICE) # (Batch, C, N, Window)
    train_Y = torch.stack(Y_list).to(DEVICE) # (Batch, N)
    
    # Modelo High-IQ
    model = ShallowGAT(num_nodes=N, in_channels=C_ext, time_steps=WINDOW, dropout=0.3, heads=16).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = BinaryFocalRankingLoss(alpha=0.7, gamma=2.5, ranking_weight=7.0)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Batching Loop
        num_batches = len(train_X) // BATCH_SIZE
        for b in range(num_batches):
            idx_start = b * BATCH_SIZE
            idx_end = idx_start + BATCH_SIZE
            
            bx = train_X[idx_start:idx_end]
            by = train_Y[idx_start:idx_end]
            
            # Injeta slots vazios (Canal 38, 39)
            bx_input = torch.zeros((BATCH_SIZE, C_ext, N, WINDOW)).to(DEVICE)
            bx_input[:, :C, :, :] = bx
            
            optimizer.zero_grad()
            pred = model(bx_input, [adj_geo, adj_conf]).squeeze()
            
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            if b % 5 == 0:
                logging.info(f"   E{epoch+1} | Batch {b}/{num_batches} | Loss: {loss.item():.4f}")
        
        # Simulação de Validação Rápida (no próprio subset)
        model.eval()
        with torch.no_grad():
            v_idx = np.random.randint(0, len(train_X), 20)
            vx = train_X[v_idx]
            vy = train_Y[v_idx]
            vx_input = torch.zeros((20, C_ext, N, WINDOW)).to(DEVICE)
            vx_input[:, :C, :, :] = vx
            v_pred = model(vx_input, [adj_geo, adj_conf]).squeeze()
            
            # P@10 simplificado
            hits = 0
            for i in range(20):
                n_real = (vy[i] > 0).sum().item()
                if n_real > 0:
                    _, t_idx = torch.topk(vy[i], min(10, n_real))
                    _, p_idx = torch.topk(v_pred[i], min(10, n_real))
                    hits += len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / min(10, n_real)
            
            avg_p10 = hits / 20
            logging.info(f"🏁 Época {epoch+1}/{EPOCHS} | Loss Médio: {epoch_loss/num_batches:.4f} | Est. P@10: {avg_p10*100:.2f}%")

    out_path = os.path.join(ROOT_DIR, "models", "tests", "fast_spike_turbo.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logging.info(f"💾 Spike Turbo salvo em {out_path}")

if __name__ == "__main__":
    run_fast_spike_v2()

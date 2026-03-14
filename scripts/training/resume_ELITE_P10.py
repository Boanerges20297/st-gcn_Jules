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
import json
import gc

# Adicionar raiz ao path para imports
sys.path.append(os.getcwd())
try:
    from src.core.architectures import DeepSTGAT_64
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src', 'core'))
    from architectures import DeepSTGAT_64

# Configuração de Logging focado em RESUME ELITE P10
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_ELITE_P10.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- HIPERPARÂMETROS IGUAIS AO ORIGINAL ---
EPOCHS = 120 
LR_MIN = 0.0001
LR_MAX = 0.018 
WINDOW = 90
DROPOUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_ACCUMULATION_STEPS = 8

# Época de onde vamos retomar (vinda dos logs)
START_EPOCH = 26 

def load_processed_data(region_key):
    path = f'data/processed/processed_{region_key}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

class ContrastiveTopKLoss(nn.Module):
    def __init__(self, k=10, margin=2.5):
        super().__init__()
        self.k = k
        self.margin = margin

    def forward(self, pred, target):
        k_eff = min(self.k, target.size(0))
        _, topk_indices = torch.topk(target, k_eff)
        
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[topk_indices] = True
        
        hotspot_scores = pred[mask]
        background_scores = pred[~mask]
        
        bg_mean = background_scores.mean()
        loss = F.relu(self.margin - (hotspot_scores - bg_mean)).mean()
        reg_penalty = 0.01 * torch.norm(pred, 2)
        
        return loss + reg_penalty

def train_region(region_key):
    logging.info(f"\n" + "="*60)
    logging.info(f"🔄 RETOMANDO TREINO ELITE P10: {region_key.upper()} (De Época {START_EPOCH+1})")
    logging.info("="*60)
    
    data = load_processed_data(region_key)
    features = data['node_features']
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    
    N, T_total, C = features.shape
    features_norm = features.copy()
    for c in range(C):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features[:, :, c] - m) / s

    X, Y = [], []
    for t in range(WINDOW, T_total - 7):
        x = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y = torch.tensor(features[:, t:t+7, 0].sum(axis=1), dtype=torch.float32)
        X.append(x)
        Y.append(y)
    
    split = int(len(X) * 0.8)
    train_X, train_Y = X[:split], Y[:split]
    val_X, val_Y = X[split:], Y[split:]
    
    model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=DROPOUT).to(DEVICE)
    
    # CARREGAR CHECKPOINT
    checkpoint_path = f'models/test/{region_key}_model_ELITE.pth'
    if os.path.exists(checkpoint_path):
        logging.info(f"📍 Carregando pesos de: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logging.warning(f"⚠️ Checkpoint {checkpoint_path} não encontrado! Iniciando do zero para esta região.")
        if region_key == 'fortaleza': # Crítico
             return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MIN, weight_decay=5e-2)
    
    k_val = 5 if region_key == 'rmf' else 10
    criterion = ContrastiveTopKLoss(k=k_val, margin=2.5)
    
    steps_per_epoch = (len(train_X) // GRADIENT_ACCUMULATION_STEPS) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=steps_per_epoch, 
        epochs=EPOCHS, pct_start=0.2
    )
    
    # AVANÇAR SCHEDULER ATÉ O PONTO DE INTERRUPÇÃO
    total_steps_to_skip = START_EPOCH * steps_per_epoch
    logging.info(f"⏭️ Avançando Scheduler em {total_steps_to_skip} steps...")
    for _ in range(total_steps_to_skip):
        scheduler.step()

    # Carregar histórico anterior para não perder o JSON de métricas
    metrics_history = []
    metrics_path = f'logs/metrics_{region_key}_ELITE.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_history = json.load(f)
            # Garantir que removemos entradas duplicadas se houver overlap
            metrics_history = [m for m in metrics_history if m['epoch'] <= START_EPOCH]

    best_p10 = max([m['p10'] for m in metrics_history]) if metrics_history else 0.0
    logging.info(f"📈 Melhor P@10 registrado: {best_p10*100:.1f}%")

    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_grads = []
        
        indices = list(range(len(train_X)))
        random.seed(epoch) # Determinismo parcial por época
        random.shuffle(indices)
        
        optimizer.zero_grad()
        for i, idx in enumerate(indices):
            pred = model(train_X[idx].to(DEVICE), [adj_geo, adj_conf]).squeeze()
            loss = criterion(pred, train_Y[idx].to(DEVICE)) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1. / 2)
                epoch_grads.append(grad_norm)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                optimizer.zero_grad()
                
                step_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                epoch_loss += step_loss
                
                current_step = (i + 1) // GRADIENT_ACCUMULATION_STEPS
                total_steps = len(train_X) // GRADIENT_ACCUMULATION_STEPS
                
                if current_step % 20 == 0:
                    with torch.no_grad():
                        _, t_idx = torch.topk(train_Y[idx].to(DEVICE), min(k_val, train_Y[idx].size(0)))
                        _, p_idx = torch.topk(pred, min(k_val, pred.size(0)))
                        batch_p = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / k_val
                    logging.info(f"   -> Step {current_step}/{total_steps} | LR: {current_lr:.5f} | Loss: {step_loss:.4f} | Grad: {grad_norm:.4f} | P@{k_val}: {batch_p*100:.1f}%")

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        # Validação
        model.eval()
        p_list = []
        with torch.no_grad():
            for vx, vy in zip(val_X, val_Y):
                vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                if vy.sum() > 0:
                    _, t_idx = torch.topk(vy, min(k_val, vy.size(0)))
                    _, p_idx = torch.topk(vpred, min(k_val, vpred.size(0)))
                    p_score = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / k_val
                    p_list.append(p_score)
        
        avg_p = np.mean(p_list) if p_list else 0
        avg_loss = epoch_loss / total_steps
        avg_grad = np.mean(epoch_grads) if epoch_grads else 0
        
        metrics_history.append({
            'epoch': epoch + 1, 'loss': float(avg_loss), 'p10': float(avg_p), 'grad': float(avg_grad)
        })
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        
        logging.info(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Grad Avg: {avg_grad:.4f} | P@{k_val}: {avg_p*100:.1f}%")
        
        if avg_p > best_p10:
            best_p10 = avg_p
            torch.save({'model_state_dict': model.state_dict(), 'config': {'window': WINDOW, 'nodes': N}}, f'models/test/{region_key}_model_ELITE.pth')
            logging.info(f"⭐ Novo Recorde: {best_p10*100:.1f}%")

def main():
    # Focar primeiro em Fortaleza (mais crítico)
    for reg in ['fortaleza', 'rmf', 'interior']:
        train_region(reg)

if __name__ == "__main__":
    main()

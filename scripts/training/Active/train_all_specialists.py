"""
train_all_specialists.py — Script oficial de retreino (FOCO CVLI - HONESTY PARADIGM).
Versão 2026-05-21: Janela 14d, MemPalace Universal, Honesty Constraint.
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import random
import gc
import subprocess
import re
from datetime import datetime

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64, DeepSTGAT_80, ShallowGAT
    from training_vault import TrainingVault
except ImportError:
    from src.core.architectures import DeepSTGAT_64, DeepSTGAT_80, ShallowGAT
    from src.core.training_vault import TrainingVault

# Configuração de Log
log_file = os.path.join(ROOT_DIR, 'logs', 'training_ALL_SPECIALISTS.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREDICT_HORIZON = 14

REGION_CONFIGS = {
    'fortaleza': dict(
        window=14, lr=3e-4, epochs=30, patience=15, dropout=0.30, margin=1.0,
        k_eval=10, use_momentum=True, grad_accum=6,
        output_name='fortaleza_model_active.pth',
        focal_alpha=0.70, focal_gamma=2.5, ranking_weight=20.0,
        scheduler='cosine_restarts', cosine_T0=10, cosine_Tmult=2, eta_min=1e-6
    ),
    'rmf': dict(
        window=14,  lr=0.001, epochs=30, patience=15, dropout=0.5, margin=1.5,
        k_eval=5,  use_momentum=True, grad_accum=8,
        output_name='rmf_model.pth',
        focal_alpha=0.50, focal_gamma=2.0, ranking_weight=10.0,
        scheduler='onecycle'
    ),
    'interior': dict(
        window=14, lr=0.001, epochs=30, patience=15, dropout=0.3, margin=1.0,
        k_eval=10, use_momentum=True,  grad_accum=4,
        output_name='interior_model.pth',
        focal_alpha=0.40, focal_gamma=2.0, ranking_weight=15.0,
        scheduler='onecycle'
    ),
}

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(adj)

def build_momentum_features(features):
    N, T, _ = features.shape
    momentum_feat = np.zeros((N, T, 4))
    cold_streak = np.zeros(N)
    for t in range(60, T):
        crimes = features[:, t, 0]
        cold_streak = np.where(crimes > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
    return momentum_feat

class BinaryFocalRankingLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ranking_weight=1.0):
        super().__init__()
        self.alpha, self.gamma, self.ranking_weight = alpha, gamma, ranking_weight
    def forward(self, pred, target, cold_streak_signal=None):
        target_bin = (target > 0).float()
        probs = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target_bin, reduction='none')
        p_t = probs * target_bin + (1 - probs) * (1 - target_bin)
        focal_loss = (self.alpha * (1 - p_t)**self.gamma * bce_loss).mean()
        rank_loss = F.mse_loss(pred[target_bin>0], target[target_bin>0]) if target_bin.sum()>0 else 0.0
        
        # ⭐ INNOVATION 1: Indecision Penalty (O "Dedo Apontado")
        # Pune o modelo se ele for "covarde" e deixar os scores muito flat (sem destaque pros top 10)
        top_vals, _ = torch.topk(pred, min(10, len(pred)))
        gap = top_vals.mean() - pred.mean()
        indecision_penalty = torch.exp(-gap) # Se o GAP for pequeno, a loss explode
        
        honesty_penalty = 0.0
        if cold_streak_signal is not None:
            calmness = torch.clamp(-cold_streak_signal / 30.0, 0, 1)
            honesty_penalty = (calmness * torch.relu(pred)).mean()
        return focal_loss + self.ranking_weight * rank_loss + 2.0 * honesty_penalty + 1.5 * indecision_penalty + 0.01 * torch.norm(pred, 2)

class SpecialistTrainer:
    def __init__(self, region_key):
        self.cfg = REGION_CONFIGS[region_key]
        self.region_key = region_key
        self.vault = None

    def train(self):
        logging.info(f"\n🚀 ESPECIALISTA: {self.region_key.upper()} (SENTINELA V4 - HONESTY)")
        path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{self.region_key}.pkl')
        with open(path, 'rb') as f: data = pickle.load(f)
        nf, adj_geo_np, adj_conf_np = data['node_features'], data['adj_geo'], data['adj_conflict']
        N, T, C_base = nf.shape
        adj_geo = torch.tensor(normalize_adj(adj_geo_np), dtype=torch.float32).to(DEVICE)
        adj_conf = torch.tensor(normalize_adj(adj_conf_np), dtype=torch.float32).to(DEVICE)
        
        # Momentum Features (4 canais)
        momentum_feat = build_momentum_features(nf)
        features = np.concatenate([nf, momentum_feat], axis=2) # (N, T, C_base + 4)
        C_ext = features.shape[2]
        
        self.vault = TrainingVault(N, ROOT_DIR)
        
        X_list, Y_list = [], []
        window = self.cfg['window']
        for t in range(window, T - PREDICT_HORIZON):
            # Entrada: (N, Window, C_ext)
            x_win = features[:, t-window:t, :].copy()
            # Permuta para (C_ext, N, Window)
            x = torch.tensor(x_win, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            # Target: soma dos crimes no horizonte de previsão para cada nó
            y = torch.tensor(nf[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
            
            # Garante 41 canais para todas as regiões (37 base + 4 momentum ou similar)
            if x.shape[1] < 41:
                padding = torch.zeros((1, 41 - x.shape[1], N, window))
                x = torch.cat([x, padding], dim=1)
            elif x.shape[1] > 41:
                x = x[:, :41, :, :]
                
            X_list.append(x)
            Y_list.append(y)

        split = int(len(X_list) * 0.85)
        train_X, train_Y = X_list[:split], Y_list[:split]
        val_X, val_Y = X_list[split:], Y_list[split:]

        model = ShallowGAT(num_nodes=N, in_channels=41, time_steps=window, dropout=self.cfg['dropout']).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg['lr'], weight_decay=0.005)
        criterion = BinaryFocalRankingLoss(alpha=self.cfg['focal_alpha'], gamma=self.cfg['focal_gamma'], ranking_weight=self.cfg['ranking_weight'])
        
        if self.cfg.get('scheduler') == 'cosine_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.cfg['cosine_T0'], T_mult=self.cfg['cosine_Tmult'], eta_min=self.cfg['eta_min'])
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg['lr'], steps_per_epoch=len(train_X)//self.cfg['grad_accum']+1, epochs=self.cfg['epochs'])

        best_pk, no_improve = 0.0, 0
        for epoch in range(self.cfg['epochs']):
            model.train(); self.vault.clear_epoch(); epoch_loss = 0.0
            idx_list = list(range(len(train_X))); random.shuffle(idx_list)
            for step, idx in enumerate(idx_list):
                xi = train_X[idx].to(DEVICE)
                if random.random() > 0.2:
                    xi[:, 37, :, :] = torch.tensor(self.vault.get_memory_vector(), dtype=torch.float32).to(DEVICE).view(1, 1, N, 1).expand(-1, -1, -1, window)
                
                cold_streak = xi[0, 36, :, -1]
                pred = model(xi, [adj_geo, adj_conf]).squeeze()
                loss = criterion(pred, train_Y[idx].to(DEVICE), cold_streak_signal=cold_streak) / self.cfg['grad_accum']
                loss.backward()
                if (step+1) % self.cfg['grad_accum'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad()
                    if self.cfg.get('scheduler') != 'cosine_restarts': scheduler.step()
                    epoch_loss += loss.item() * self.cfg['grad_accum']
            
            model.eval(); pk10 = []
            with torch.no_grad():
                for vx, vy in zip(val_X, val_Y):
                    if (vy>0).sum() > 0:
                        vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                        k = min(10, (vy>0).sum().item(), N)
                        _, t_idx = torch.topk(vy.to(DEVICE), k); _, p_idx = torch.topk(vpred, 10)
                        pk10.append(len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / k)
            
            avg_p = np.mean(pk10) if pk10 else 0.0
            logging.info(f"E{epoch+1:03d} | Val P@10: {avg_p*100:.2f}% | Loss: {epoch_loss/len(train_X):.4f}")
            if avg_p > best_pk:
                best_pk = avg_p; no_improve = 0
                torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'models', 'active', self.cfg['output_name']))
                logging.info(f"💎 NOVO RECORDE: {best_pk*100:.2f}%")
            else:
                no_improve += 1
                if no_improve >= self.cfg['patience']: break
            if self.cfg.get('scheduler') == 'cosine_restarts': scheduler.step()

def main():
    for region in REGION_CONFIGS.keys():
        try: SpecialistTrainer(region).train()
        except Exception as e: logging.error(f"❌ Erro em {region}: {e}")

if __name__ == "__main__": main()
 if self.cfg.get('scheduler') == 'cosine_restarts': scheduler.step()

def main():
    for region in REGION_CONFIGS.keys():
        try: SpecialistTrainer(region).train()
        except Exception as e: logging.error(f"❌ Erro em {region}: {e}")

if __name__ == "__main__": main()

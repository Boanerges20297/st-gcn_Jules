import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
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

# Configuração de Logging Extremo
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_RETRAIN_64.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- HIPERPARÂMETROS SOLICITADOS ---
EPOCHS = 120 
LR_MAX = 0.01 # Taxa de aprendizado solicitada (0.01)
WINDOW = 120 # Mantemos 120 dias para o Multi-Scale Momentum
DROPOUT = 0.3 # Dropout solicitado (0.3)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_ACCUMULATION_STEPS = 32 # Equivalente ao Batch Size = 32 solicitado
REGION = 'fortaleza'

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
    def __init__(self, k=10, margin=1.0):
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

def train_retrain_64():
    logging.info("\n" + "="*80)
    logging.info(f"🚀 INICIANDO RETREINO DETALHADO - DeepSTGAT_64 (TENTATIVA 46 - BASE TOTAL + SPLIT ALEATÓRIO)")
    logging.info(f"⚙️ PARAMETROS: LR_MAX={LR_MAX} | BATCH={GRADIENT_ACCUMULATION_STEPS} | DROPOUT={DROPOUT} | MARGIN=1.0")
    logging.info("="*80)
    
    data = load_processed_data(REGION)
    
    # --- BASE TOTAL (2022-2026) ---
    features = data['node_features']
    logging.info(f"📅 Treinando com a Base Total: {features.shape[1]} dias processados.")
    
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    
    N, T_total, C = features.shape
    
    # --- ENGENHARIA DE MULTI-SCALE MOMENTUM (33 Canais - Quente e Frio) ---
    momentum_feat = np.zeros((N, T_total, 4))
    cold_streak = np.zeros(N)
    for t in range(60, T_total):
        recent_7 = features[:, t-7:t, 0].sum(axis=1)
        past_7 = features[:, t-14:t-7, 0].sum(axis=1)
        momentum_feat[:, t, 0] = recent_7 - past_7
        recent_14 = features[:, t-14:t, 0].sum(axis=1)
        past_14 = features[:, t-28:t-14, 0].sum(axis=1)
        momentum_feat[:, t, 1] = recent_14 - past_14
        recent_30 = features[:, t-30:t, 0].sum(axis=1)
        past_30 = features[:, t-60:t-30, 0].sum(axis=1)
        momentum_feat[:, t, 2] = recent_30 - past_30
        crimes_today = features[:, t, 0]
        cold_streak = np.where(crimes_today > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
    
    features_extended = np.concatenate([features, momentum_feat], axis=2)
    C_ext = features_extended.shape[2]

    # Dados brutos — sem normalização para preservar picos de criminalidade

    X, Y = [], []
    for t in range(WINDOW, T_total - 7):
        x = torch.tensor(features_extended[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y = torch.tensor(features[:, t:t+7, 0].sum(axis=1), dtype=torch.float32)
        X.append(x)
        Y.append(y)
    
    # --- SPLIT ALEATÓRIO (SHUFFLE) PARA VALIDAÇÃO INÉDITA ---
    indices = list(range(len(X)))
    random.seed(42) # Semente para reprodutibilidade do split
    random.shuffle(indices)
    
    split = int(len(X) * 0.85)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_X = [X[i] for i in train_indices]
    train_Y = [Y[i] for i in train_indices]
    val_X = [X[i] for i in val_indices]
    val_Y = [Y[i] for i in val_indices]
    
    logging.info(f"🔀 Split Aleatório Concluído: {len(train_X)} amostras de treino | {len(val_X)} amostras de validação inédita.")
    
    model = DeepSTGAT_64(num_nodes=N, in_channels=C_ext, time_steps=WINDOW, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    criterion = ContrastiveTopKLoss(k=10, margin=1.0)
    
    steps_per_epoch = (len(train_X) // GRADIENT_ACCUMULATION_STEPS) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=steps_per_epoch, 
        epochs=EPOCHS, pct_start=0.2
    )
    
    best_p10 = 0.0
    os.makedirs('models/active', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_grads = []
        
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        
        optimizer.zero_grad()
        for i, idx in enumerate(indices):
            pred = model(train_X[idx].to(DEVICE), [adj_geo, adj_conf]).squeeze()
            loss = criterion(pred, train_Y[idx].to(DEVICE)) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Fechamento do Batch de 32
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Log Extremo do Gradiente antes do Clipping
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1. / 2)
                epoch_grads.append(grad_norm)
                
                # Clipping severo para lidar com LR=0.05
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                optimizer.zero_grad()
                
                step_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                epoch_loss += step_loss
                
                current_step = (i + 1) // GRADIENT_ACCUMULATION_STEPS
                
                # Log detalhado a cada step executado
                with torch.no_grad():
                    _, t_idx = torch.topk(train_Y[idx].to(DEVICE), 10)
                    _, p_idx = torch.topk(pred, 10)
                    batch_p = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / 10.0
                logging.info(f"E{epoch+1:03d} | Batch {current_step:03d} | LR: {current_lr:.5f} | Loss: {step_loss:.4f} | GradNorm: {grad_norm:.4f} | Batch P@10: {batch_p*100:.1f}%")

        # Validação
        model.eval()
        p_list = []
        with torch.no_grad():
            for vx, vy in zip(val_X, val_Y):
                if vy.sum() > 0:
                    vpred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
                    _, t_idx = torch.topk(vy, 10)
                    _, p_idx = torch.topk(vpred, 10)
                    p_score = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy())) / 10.0
                    p_list.append(p_score)
        
        avg_p = np.mean(p_list) if p_list else 0
        avg_loss = epoch_loss / steps_per_epoch
        avg_grad = np.mean(epoch_grads) if epoch_grads else 0
        
        current_best = max(best_p10, avg_p)
        logging.info(f"\n---> FIM DA ÉPOCA {epoch+1} | Val P@10: {avg_p*100:.4f}% | Loss Média: {avg_loss:.4f} | Grad Médio: {avg_grad:.4f} | Recorde Geral: {current_best*100:.4f}% <---\n")
        
        if avg_p > best_p10:
            best_p10 = avg_p
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'window': WINDOW, 'nodes': N, 'arch': 'DeepSTGAT_64', 'in_channels': C_ext}
            }, 'models/active/fortaleza_retrain_64.pth')
            logging.info(f"💎 NOVO RECORDE RETREINO 64: {best_p10*100:.1f}%")

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    train_retrain_64()

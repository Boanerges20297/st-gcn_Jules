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
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.core.architectures import TemperatureExpertGAT
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src', 'core'))
    from architectures import TemperatureExpertGAT

# Configuração de Logging detalhado
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_regime_experts.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- RETORNO AO BASICO QUE FUNCIONA ---
EPOCHS = 150
LR = 0.001
DROPOUT = 0.4
RANKING_WEIGHT = 5.0
BATCH_SIZE = 32

# INTEL-BIAS ATIVO
FACTION_PRIORITY = {
    'CV': 2.0,
    'MASSA': 2.0,
    'GDE': 1.0,
    'PCC': 1.0,
    'TCP': 1.0,
    'NEUTRO': 1.0
}

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def load_data():
    path = os.path.join(ROOT, 'data', 'processed', 'processed_fortaleza.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def main():
    data = load_data()
    features = data['node_features']
    dates = pd.to_datetime(data['dates'])
    nodes_gdf = data['nodes_gdf']
    
    # --- FILTRO ESTRATEGICO: APENAS DIAS QUENTES (> 3 crimes/dia na cidade) ---
    # O modelo vira um "Detector de Crise Diária". Ignoramos dias calmos (ruido).
    # Focando em sazonalidade (Fins de semana intensos)
    
    daily_sums = features[:, :, 0].sum(axis=0) # (TimeSteps,)
    mask_hot = daily_sums > 3
    
    # Aplicar filtro
    features = features[:, mask_hot, :]
    dates = dates[mask_hot]
    
    logging.info(f"🚀 INICIANDO TREINO DE 'PENEIRA QUENTE' (DIARIO > 3 CRIMES)")
    logging.info(f"   Amostras Criticas: {features.shape[1]} dias | Foco: Detectar Picos")

    WINDOW, PREDICT_HORIZON = 30, 7
    N, T_total, C = features.shape
    
    model = TemperatureExpertGAT(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=DROPOUT).to(DEVICE)
    model.apply(initialize_weights) # RESET TOTAL
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    node_factions = nodes_gdf['faction'].values
    faction_weights = torch.tensor([FACTION_PRIORITY.get(f, 1.0) for f in node_factions], dtype=torch.float32).to(DEVICE)
    
    X_list, y_list = [], []
    adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)
    features_norm = features.copy()
    for c in range(C):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features_norm[:, :, c] - m) / s

    for t in range(WINDOW, T_total - PREDICT_HORIZON):
        x_t = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
        y_target = y_raw + (0.3 * torch.matmul(adj_dense, y_raw))
        if y_target.max() > 0: y_target = y_target / y_target.max()
        X_list.append(x_t)
        y_list.append(y_target.unsqueeze(0))

    split = int(len(X_list) * 0.8)
    train_X, train_y = X_list[:split], y_list[:split]
    val_X, val_y = X_list[split:], y_list[split:]

    adj_geo_t = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf_t = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)

    def criterion(pred, target):
        loss_reg = (faction_weights * F.smooth_l1_loss(pred, target, reduction='none')).mean()
        pred_var = torch.var(pred)
        variance_penalty = F.relu(0.01 - pred_var) * 100.0
        k = 30
        top_val, top_idx = torch.topk(target, min(k, len(target)))
        if top_val.sum() == 0: return loss_reg + 0.1 * variance_penalty
        num_neg = 50
        neg_idx = torch.randint(0, len(target), (num_neg,), device=target.device)
        p_h, p_l = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
        t_h, t_l = target[top_idx].unsqueeze(1), target[neg_idx].unsqueeze(0)
        margin = 0.1 + (F.relu(t_h - t_l) * 0.5)
        loss_rank = (F.relu(margin - (p_h - p_l)) * (t_h > t_l).float()).sum() / (num_neg * k)
        return loss_reg + RANKING_WEIGHT * loss_rank + 0.1 * variance_penalty

    best_recall = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        
        optimizer.zero_grad()
        for i, idx in enumerate(indices):
            pred = model(train_X[idx].to(DEVICE), [adj_geo_t, adj_conf_t]).squeeze()
            target = train_y[idx].squeeze().to(DEVICE)
            loss = criterion(pred, target) / BATCH_SIZE
            loss.backward()
            epoch_loss += loss.item() * BATCH_SIZE
            
            if (i+1) % BATCH_SIZE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if (i+1) % 100 == 0:
                logging.info(f"   [Epoch {epoch+1:02d}] Step {i+1}/{len(indices)} | Loss: {loss.item()*BATCH_SIZE:.6f}")

        model.eval()
        hits_40 = []
        with torch.no_grad():
            for vx, vy in zip(val_X, val_y):
                vpred = model(vx.to(DEVICE), [adj_geo_t, adj_conf_t]).squeeze().cpu().numpy()
                vtrue = vy.squeeze().numpy()
                if vtrue.sum() == 0: continue
                top_10_true = np.argsort(-vtrue)[:10] 
                top_40_pred = np.argsort(-vpred)[:40] 
                recall = len(set(top_10_true) & set(top_40_pred)) / len(top_10_true)
                hits_40.append(recall)
        
        avg_recall = np.mean(hits_40) if hits_40 else 0
        logging.info(f"📈 EPOCH {epoch+1:02d} FINAL | Loss: {epoch_loss/len(indices):.6f} | Recall@40: {avg_recall*100:.1f}%")
        
        path = os.path.join(ROOT, 'models', 'test', 'ranking', 'fortaleza_expert_universal.pth')
        if avg_recall > best_recall:
            best_recall = avg_recall
            torch.save({'model_state_dict': model.state_dict(), 'recall': avg_recall}, path)
            logging.info(f"🏆 NOVO RECORDE UNIVERSAL: Recall@40 = {avg_recall*100:.1f}%")

if __name__ == "__main__":
    main()

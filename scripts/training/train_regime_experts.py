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
    
    # --- FILTRO ESPACIAL: APENAS BAIRROS ATIVOS (> 1 CVLI/MÊS) ---
    # Remove ruído de bairros nobres/seguros (Aldeota, Meireles, etc.)
    # Foco total em territórios de conflito.
    
    total_cvli_per_node = features[:, :, 0].sum(axis=1) # (Nodes,)
    num_months = features.shape[1] / 30.0
    threshold_total = num_months * 1.0 # > 1 por mês
    
    active_mask = total_cvli_per_node > threshold_total
    active_indices = np.where(active_mask)[0]

    logging.info(f"🏙️ FILTRO ESPACIAL: {len(active_indices)} bairros ativos (de {len(nodes_gdf)})")

    # Filtrar tensores e GDF
    features = features[active_indices, :, :]
    nodes_gdf = nodes_gdf.iloc[active_indices].reset_index(drop=True)

    # Recalcular matrizes de adjacência para o subgrafo
    coords = np.array(list(zip(nodes_gdf.geometry.x, nodes_gdf.geometry.y)))
    from scipy.spatial.distance import cdist
    adj_geo_new = (cdist(coords, coords) <= 3000).astype(float)
    adj_conf_new = np.eye(len(nodes_gdf)) # Placeholder, idealmente recalcularia conflitos se disponível

    # Atualizar dicionário de dados localmente
    data['adj_geo'] = adj_geo_new
    data['adj_conflict'] = adj_conf_new

    # --- FILTRO TEMPORAL: APENAS DIAS QUENTES (> 3 crimes/dia NA REDE FILTRADA) ---
    # O modelo vira um "Detector de Crise Diária". Ignoramos dias calmos (ruido).

    daily_sums = features[:, :, 0].sum(axis=0) # (TimeSteps,)
    mask_hot = daily_sums > 2 # Reduzido de 3 para 2 pois removemos 102 bairros

    # Aplicar filtro temporal
    features = features[:, mask_hot, :]
    dates = dates[mask_hot]
    
    logging.info(f"🚀 INICIANDO TREINO FOCADO (19 Bairros | Diario > 2 Crimes)")
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
        prec_5_list = []
        prec_10_list = []
        with torch.no_grad():
            for vx, vy in zip(val_X, val_y):
                vpred = model(vx.to(DEVICE), [adj_geo_t, adj_conf_t]).squeeze().cpu().numpy()
                vtrue = vy.squeeze().numpy()
                if vtrue.sum() == 0: continue

                # Ground Truth: Top 5 e Top 10 reais
                k_true = min(10, len(vtrue))
                top_k_true = set(np.argsort(-vtrue)[:k_true])

                # Predictions
                top_5_pred = set(np.argsort(-vpred)[:5])
                top_10_pred = set(np.argsort(-vpred)[:10])

                # Precision@K: Quantos dos K previstos estavam no Top K real?
                p5 = len(top_5_pred & top_k_true) / 5
                p10 = len(top_10_pred & top_k_true) / 10

                prec_5_list.append(p5)
                prec_10_list.append(p10)

        avg_p5 = np.mean(prec_5_list) if prec_5_list else 0
        avg_p10 = np.mean(prec_10_list) if prec_10_list else 0
        
        logging.info(f"📈 EPOCH {epoch+1:02d} | Loss: {epoch_loss/len(indices):.4f} | P@5: {avg_p5*100:.1f}% | P@10: {avg_p10*100:.1f}%")
        
        path = os.path.join(ROOT, 'models', 'test', 'ranking', 'fortaleza_expert_universal.pth')
        # Salvar se P@10 melhorar
        if avg_p10 > best_recall:
            best_recall = avg_p10
            torch.save({'model_state_dict': model.state_dict(), 'p10': avg_p10}, path)
            logging.info(f"🏆 NOVO RECORDE: P@10 = {avg_p10*100:.1f}%")

if __name__ == "__main__":
    main()

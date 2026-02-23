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
EPOCHS = 200
LR = 0.001
DROPOUT = 0.3
RANKING_WEIGHT = 10.0
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
    
    # --- FILTRO TEMPORAL: 2024 a 2026 ---
    # Foco em dados recentes conforme solicitado.
    mask_date = (dates >= pd.Timestamp('2024-01-01')) & (dates <= pd.Timestamp('2026-12-31'))
    features = features[:, mask_date, :]
    dates = dates[mask_date]
    logging.info(f"📅 FILTRO TEMPORAL: {len(dates)} dias (2024-2026)")

    # --- FILTRO ESPACIAL: LEAGUE OF VIOLENCE (TOP 50) ---
    # Foco absoluto nos 50 bairros mais violentos.
    # Garante densidade de crime para o modelo aprender padrões reais.

    total_cvli_per_node = features[:, :, 0].sum(axis=1) # (Nodes,)
    top_k_indices = np.argsort(-total_cvli_per_node)[:30]

    logging.info(f"🏙️ FILTRO ESPACIAL: Top 30 Bairros Mais Violentos Selecionados (League of Violence Elite)")

    # Filtrar tensores e GDF
    features = features[top_k_indices, :, :]
    nodes_gdf = nodes_gdf.iloc[top_k_indices].reset_index(drop=True)

    # Recalcular matrizes de adjacência para o subgrafo
    coords = np.array(list(zip(nodes_gdf.geometry.x, nodes_gdf.geometry.y)))
    from scipy.spatial.distance import cdist
    adj_geo_new = (cdist(coords, coords) <= 3000).astype(float)
    adj_conf_new = np.eye(len(nodes_gdf))

    data['adj_geo'] = adj_geo_new
    data['adj_conflict'] = adj_conf_new

    # --- FILTRO DE INTENSIDADE DIÁRIA ---
    daily_sums = features[:, :, 0].sum(axis=0)
    mask_hot = daily_sums > 3 # Voltar para > 3 pois temos mais bairros agora

    features = features[:, mask_hot, :]
    dates = dates[mask_hot]
    
    logging.info(f"🚀 INICIANDO TREINO (2024-2026 | Diario > 3 Crimes)")
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
        # 1. Hotspot Weighted Regression (Foco nos picos)
        # Se target > 0, erro pesa 5x mais.
        weights = torch.ones_like(target)
        weights[target > 0] = 5.0
        loss_reg = (weights * F.smooth_l1_loss(pred, target, reduction='none')).mean()

        # 2. Ranking Loss (Pairwise Margin)
        # Garante que Hotspots fiquem acima de Non-Hotspots
        k = 10 # Foco no Top 10
        top_val, top_idx = torch.topk(target, min(k, len(target)))

        if top_val.sum() == 0: return loss_reg

        # Amostrar negativos (quem tem menos crime que o top k)
        # Simplificacao: comparar Top K contra o resto
        mask_top = torch.zeros_like(target, dtype=torch.bool)
        mask_top[top_idx] = True
        neg_idx = torch.where(~mask_top)[0]

        if len(neg_idx) > 0:
            # Selecionar alguns negativos aleatorios
            perm = torch.randperm(len(neg_idx))[:20]
            neg_idx_sel = neg_idx[perm]

            p_h = pred[top_idx].unsqueeze(1) # (K, 1)
            p_l = pred[neg_idx_sel].unsqueeze(0) # (1, M)
            t_h = target[top_idx].unsqueeze(1)
            t_l = target[neg_idx_sel].unsqueeze(0)

            # Margem dinamica baseada na diferenca real
            margin = 0.1 + (t_h - t_l)

            # Loss: max(0, margin - (high_pred - low_pred))
            loss_rank = F.relu(margin - (p_h - p_l)).mean()
        else:
            loss_rank = 0.0

        return loss_reg + (RANKING_WEIGHT * loss_rank)

    best_recall = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        
        optimizer.zero_grad()
        for i, idx in enumerate(indices):
            # Input X shape: (Batch, Channels, Nodes, Time)
            batch_x = train_X[idx].to(DEVICE)

            # Persistence: Pegar o ultimo dia de CVLI (Canal 0) para cada no
            # batch_x eh (1, 29, 30, 30) -> (Batch, Ch, Node, Time) ???
            # O script cria: features_norm[:, t-WINDOW:t, :] -> (N, T, C)
            # permute(2, 0, 1) -> (C, N, T)
            # unsqueeze(0) -> (1, C, N, T)
            # Entao Time eh a ultima dimensao.
            # Ultimo dia esta em -1.

            # persistence = batch_x[:, 0, :, -1].squeeze() # (Nodes,)
            # Mas o modelo ja aprende isso. Vamos forcar o residuo.

            pred_raw = model(batch_x, [adj_geo_t, adj_conf_t]).squeeze()

            # Hybrid Prediction: 0.5 * Model + 0.5 * Persistence
            # persistence = batch_x[0, 0, :, -1]
            # pred = pred_raw + persistence

            pred = pred_raw # Manter puro por enquanto, arquitetura ja tem conexoes

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
        prec_10_list = []
        prec_20_list = []
        with torch.no_grad():
            for vx, vy in zip(val_X, val_y):
                # Persistence bias no teste tambem?
                # persistence = vx[0, 0, :, -1].to(DEVICE)

                vpred = model(vx.to(DEVICE), [adj_geo_t, adj_conf_t]).squeeze().cpu().numpy()
                # vpred = vpred + persistence.cpu().numpy()

                vtrue = vy.squeeze().numpy()
                if vtrue.sum() == 0: continue

                # Ground Truth: Top 20 reais (ou total se menor)
                k_true_20 = min(20, len(vtrue))
                top_20_true = set(np.argsort(-vtrue)[:k_true_20])

                # Ground Truth: Top 10 reais
                k_true_10 = min(10, len(vtrue))
                top_10_true = set(np.argsort(-vtrue)[:k_true_10])

                # Predictions
                top_10_pred = set(np.argsort(-vpred)[:10])
                top_20_pred = set(np.argsort(-vpred)[:20])

                # Precision@K: Quantos dos K previstos estavam no Top K real?
                p10 = len(top_10_pred & top_10_true) / 10
                p20 = len(top_20_pred & top_20_true) / 20

                prec_10_list.append(p10)
                prec_20_list.append(p20)

        avg_p10 = np.mean(prec_10_list) if prec_10_list else 0
        avg_p20 = np.mean(prec_20_list) if prec_20_list else 0
        
        logging.info(f"📈 EPOCH {epoch+1:02d} | Loss: {epoch_loss/len(indices):.4f} | P@10: {avg_p10*100:.1f}% | P@20: {avg_p20*100:.1f}%")
        
        path = os.path.join(ROOT, 'models', 'test', 'ranking', 'fortaleza_expert_universal.pth')
        # Salvar se P@20 melhorar (solicitacao do usuario: foque nos top 20)
        if avg_p20 > best_recall:
            best_recall = avg_p20
            torch.save({'model_state_dict': model.state_dict(), 'p20': avg_p20}, path)
            logging.info(f"🏆 NOVO RECORDE: P@20 = {avg_p20*100:.1f}%")

if __name__ == "__main__":
    main()

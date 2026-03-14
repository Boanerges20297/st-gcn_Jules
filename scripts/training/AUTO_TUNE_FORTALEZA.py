import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import random
import gc

sys.path.append(os.getcwd())
try:
    from src.core.architectures import DeepSTGAT_64
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src', 'core'))
    from architectures import DeepSTGAT_64

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_AUTO_TUNE.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_P10 = 0.50 # Meta absoluta: 50% P@10

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

def train_with_window(data, window_size, max_epochs=60):
    """Treina o modelo para uma janela específica e retorna o melhor P@10."""
    logging.info(f"\n{'='*50}")
    logging.info(f"⏳ INICIANDO TREINO COM JANELA RESTRITA: {window_size} DIAS")
    logging.info(f"{'='*50}")
    
    features = data['node_features']
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    
    N, T_total, C = features.shape
    features_norm = features.copy()
    for c in range(C):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features[:, :, c] - m) / s

    X, Y = [], []
    # O corte temporal define a dificuldade do problema
    for t in range(window_size, T_total - 7):
        x = torch.tensor(features_norm[:, t-window_size:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y = torch.tensor(features[:, t:t+7, 0].sum(axis=1), dtype=torch.float32)
        X.append(x)
        Y.append(y)
    
    split = int(len(X) * 0.8)
    train_X, train_Y = X[:split], Y[:split]
    val_X, val_Y = X[split:], Y[split:]
    
    # Mantemos 64 neurônios para focar puramente no ganho temporal
    model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=window_size, dropout=0.5).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
    criterion = ContrastiveTopKLoss(k=10, margin=2.5)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.015, steps_per_epoch=(len(train_X)//8)+1, epochs=max_epochs, pct_start=0.2
    )
    
    best_val_p10 = 0.0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        indices = list(range(len(train_X)))
        random.shuffle(indices)
        
        optimizer.zero_grad()
        for i, idx in enumerate(indices):
            pred = model(train_X[idx].to(DEVICE), [adj_geo, adj_conf]).squeeze()
            loss = criterion(pred, train_Y[idx].to(DEVICE)) / 8
            loss.backward()
            
            if (i + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

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
        logging.info(f"Janela {window_size}d | Época {epoch+1}/{max_epochs} | Val P@10: {avg_p*100:.1f}%")
        
        if avg_p > best_val_p10:
            best_val_p10 = avg_p
            epochs_without_improvement = 0
            # Salva o melhor modelo dessa janela
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'window': window_size, 'nodes': N, 'arch': 'DeepSTGAT_64'}
            }, f'models/active/fortaleza_auto_tune_w{window_size}.pth')
        else:
            epochs_without_improvement += 1
            
        if avg_p >= TARGET_P10:
            logging.info(f"🎯 META DE {TARGET_P10*100}% ATINGIDA com janela de {window_size} dias!")
            break
            
        # Early stopping adaptativo
        if epochs_without_improvement > 15:
            logging.info(f"⚠️ Platô detectado na janela de {window_size} dias. Interrompendo para tentar janela menor.")
            break

    return best_val_p10

def run_auto_tuner():
    logging.info("🚀 INICIANDO AUTO-TUNER DE JANELA TEMPORAL (FORTALEZA)")
    data = load_processed_data('fortaleza')
    
    # Degradação sucessiva da janela de tempo: 120 -> 90 -> 60 -> 30 dias
    windows_to_test = [120, 90, 60, 30]
    
    for w in windows_to_test:
        best_p10 = train_with_window(data, window_size=w)
        logging.info(f"📊 Resultado Final para janela {w}d: P@10 = {best_p10*100:.1f}%")
        
        if best_p10 >= TARGET_P10:
            logging.info(f"🏆 SUCESSO ABSOLUTO! O modelo validou o uso da janela de {w} dias.")
            # Promove o modelo vitorioso
            os.replace(f'models/active/fortaleza_auto_tune_w{w}.pth', 'models/active/fortaleza_model_active.pth')
            break
        else:
            logging.warning(f"📉 Janela de {w} dias não atingiu os 50%. Encolhendo a janela para o próximo ciclo...")

if __name__ == "__main__":
    run_auto_tuner()

import os
import sys
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64
except ImportError:
    from src.core.architectures import DeepSTGAT_64

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def build_momentum_features(features):
    N, T, _ = features.shape
    momentum_feat = np.zeros((N, T, 4))
    cold_streak = np.zeros(N)
    for t in range(60, T):
        r7  = features[:, t-7:t,   0].sum(axis=1)
        p7  = features[:, t-14:t-7, 0].sum(axis=1)
        momentum_feat[:, t, 0] = r7 - p7
        r14 = features[:, t-14:t,   0].sum(axis=1)
        p14 = features[:, t-28:t-14, 0].sum(axis=1)
        momentum_feat[:, t, 1] = r14 - p14
        r30 = features[:, t-30:t,   0].sum(axis=1)
        p30 = features[:, t-60:t-30, 0].sum(axis=1)
        momentum_feat[:, t, 2] = r30 - p30
        crimes = features[:, t, 0]
        cold_streak = np.where(crimes > 0, 0, cold_streak + 1)
        momentum_feat[:, t, 3] = -np.clip(cold_streak, 0, 30)
    return momentum_feat

def evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    region = 'fortaleza'
    model_path = 'models/active/legacy_torch/fortaleza_model_active.pth'
    data_path = f'data/processed/processed_{region}.pkl'

    print(f"🚀 Avaliando Modelo Blindado: {model_path}")
    
    # 1. Carregar Dados
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    nf = data['node_features']
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    N, T, _ = nf.shape

    # 2. Reconstruir Features (Canal 24 e Momentum)
    for n in range(N):
        nf[n, :, 24] = pd.Series(nf[n, :, 0]).rolling(window=7, min_periods=1).sum().values
    
    mf = build_momentum_features(nf)
    features = np.concatenate([nf, mf], axis=2)
    C_ext = features.shape[2]

    # 3. Carregar Modelo
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    window = config['window']
    model = DeepSTGAT_64(num_nodes=N, in_channels=C_ext, time_steps=window).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 4. Definir Janela de Validação (Últimos 15% - Cego)
    # Replicando a lógica do train_all_specialists.py
    X_all, Y_all = [], []
    for t in range(window, T - 14):
        x_window = features[:, t-window:t, :].copy()
        # Normalização Local Z-Score
        for c in range(C_ext):
            m = x_window[:, :, c].mean()
            s = x_window[:, :, c].std() + 1e-6
            x_window[:, :, c] = (x_window[:, :, c] - m) / s
        
        x = torch.tensor(x_window, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y = nf[:, t:t+14, 0].sum(axis=1)
        X_all.append(x)
        Y_all.append(y)

    split = int(len(X_all) * 0.85)
    val_X = X_all[split:]
    val_Y = Y_all[split:]
    
    print(f"📊 Avaliando em {len(val_X)} janelas do futuro...")

    p10_list = []
    
    with torch.no_grad():
        for vx, vy in tqdm(zip(val_X, val_Y), total=len(val_X)):
            pred = model(vx.to(DEVICE), [adj_geo, adj_conf]).squeeze()
            
            # Ground Truth Top 10
            k_eff = min(10, (vy > 0).sum())
            if k_eff == 0: continue # Ignora se não houve crime nenhum na janela (raro em Fortaleza)
            
            _, t_idx = torch.topk(torch.tensor(vy), k_eff)
            _, p_idx = torch.topk(pred, 10)
            
            hits = len(set(t_idx.cpu().numpy()) & set(p_idx.cpu().numpy()))
            p10_list.append(hits / k_eff)

    print("\n" + "="*40)
    print(f"🎯 PERFORMANCE CEGA FINAL (P@10): {np.mean(p10_list)*100:.2f}%")
    print(f"📈 Estabilidade: Std Dev: {np.std(p10_list)*100:.2f} pp")
    print(f"🏆 Melhor Janela: {np.max(p10_list)*100:.2f}%")
    print(f"📉 Pior Janela: {np.min(p10_list)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate()

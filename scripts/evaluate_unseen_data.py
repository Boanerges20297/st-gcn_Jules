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
from torch.utils.data import DataLoader, TensorDataset

# Caminhos de sistema
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))

try:
    from architectures import DeepSTGAT_64
except ImportError:
    from src.core.architectures import DeepSTGAT_64

# Configuração de Log para o Teste
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW = 90
PREDICT_HORIZON = 7
TEST_SIZE = 30 # Avaliar os últimos 30 dias inéditos

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def calculate_p_at_k(y_true, y_pred, k=10):
    if y_true.sum() == 0: return None
    t_true = np.argsort(y_true)[::-1][:k]
    t_pred = np.argsort(y_pred)[::-1][:k]
    return len(set(t_true) & set(t_pred)) / k

def evaluate_region(region_key):
    logging.info(f"\n🔍 AVALIANDO REGIONAL: {region_key.upper()}")
    
    # 1. Carregar Dados
    path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{region_key}.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)

    nf = data['node_features'] 
    adj_geo_np = data['adj_geo']
    adj_conf_np = data['adj_conflict']
    N, T, C = nf.shape

    adj_geo = torch.tensor(normalize_adj(adj_geo_np), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(adj_conf_np, dtype=torch.float32).to(DEVICE)

    # Normalização Z-Score (usando parâmetros globais para simular produção)
    f_norm = nf.copy()
    for c in range(C):
        m, s = nf[:,:,c].mean(), nf[:,:,c].std() + 1e-5
        f_norm[:,:,c] = (nf[:,:,c] - m) / s

    # 2. Isolar Janelas de Teste (Últimos TEST_SIZE dias)
    X, Y = [], []
    # Pegamos as últimas janelas que terminam no fim do dataset
    for t in range(T - TEST_SIZE, T - PREDICT_HORIZON):
        x = torch.tensor(f_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2,0,1)
        y_real = nf[:, t:t+PREDICT_HORIZON, 0].sum(axis=1) # Contagem real de crimes
        X.append(x)
        Y.append(y_real)

    X = torch.stack(X)
    Y = np.stack(Y)

    # 3. Carregar Modelo
    model_path = os.path.join(ROOT_DIR, 'models', 'active', 'legacy_torch', f'{region_key}_model.pth')
    if not os.path.exists(model_path):
        logging.error(f"❌ Modelo não encontrado em {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = DeepSTGAT_64(num_nodes=N, in_channels=C, time_steps=WINDOW).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Inferência e Cálculo de Métricas
    with torch.no_grad():
        preds = model(X.to(DEVICE), [adj_geo, adj_conf]).squeeze(-1).cpu().numpy()
    
    # Baseline: Média Histórica (Prever baseado no acumulado total de crimes por nó)
    baseline_pred = nf[:, :, 0].sum(axis=1) 
    
    metrics = {'P@5': [], 'P@10': [], 'P@20': []}
    baseline_metrics = {'P@5': [], 'P@10': [], 'P@20': []}

    for i in range(len(Y)):
        for k in [5, 10, 20]:
            if k > N: continue
            p_k = calculate_p_at_k(Y[i], preds[i], k)
            p_k_base = calculate_p_at_k(Y[i], baseline_pred, k)
            
            if p_k is not None: metrics[f'P@{k}'].append(p_k)
            if p_k_base is not None: baseline_metrics[f'P@{k}'].append(p_k_base)

    # 5. Relatório
    print(f"\n--- RESULTADOS FINAIS: {region_key.upper()} ---")
    for k in [5, 10, 20]:
        m = np.mean(metrics[f'P@{k}']) * 100 if metrics[f'P@{k}'] else 0
        b = np.mean(baseline_metrics[f'P@{k}']) * 100 if baseline_metrics[f'P@{k}'] else 0
        gain = m - b
        print(f"P@{k:02d}: {m:.1f}% | Baseline (Hist): {b:.1f}% | Ganho: {gain:+.1f}%")

def main():
    regions = ['fortaleza', 'rmf', 'interior']
    for r in regions:
        try:
            evaluate_region(r)
        except Exception as e:
            logging.error(f"Erro ao avaliar {r}: {e}")

if __name__ == "__main__":
    main()

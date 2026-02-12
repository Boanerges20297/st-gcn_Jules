import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURAÇÃO DE CAMINHOS (CORRIGIDA)
# ==============================================================================
# Pega o diretório onde o script está rodando
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Se estiver rodando da raiz (st-gcn_jules), o ROOT é o próprio diretório
# Se estiver em src/, o ROOT é um nível acima
if os.path.basename(CURRENT_DIR) == 'src':
    ROOT = os.path.dirname(CURRENT_DIR)
else:
    ROOT = CURRENT_DIR

sys.path.insert(0, ROOT)

# ==============================================================================
# IMPORTS E PARÂMETROS
# ==============================================================================
try:
    from src.model import STGCN
except ImportError:
    # Fallback se não encontrar o módulo (tenta adicionar src manualmente)
    sys.path.append(os.path.join(ROOT, 'src'))
    try:
        from model import STGCN
    except ImportError:
        print("❌ Erro Crítico: Não foi possível importar STGCN. Verifique se 'src/model.py' existe.")
        sys.exit(1)

DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
STGCN_PATH = os.path.join(ROOT, 'models', 'stgcn_model_v2.pth')
RANKING_DIR = os.path.join(ROOT, 'models', 'ranking_by_day')
SCALER_PATH = os.path.join(RANKING_DIR, 'ranking_scaler.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PESOS DA ESTRATÉGIA HÍBRIDA (90% ST-GCN / 10% Ranking)
WEIGHT_STGCN = 0.90
WEIGHT_RANKING = 0.10

# ==============================================================================
# DEFINIÇÕES LOCAIS
# ==============================================================================
class RankingModelV3(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128):
        super(RankingModelV3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze()

def compute_norm_adj(adj):
    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt).to(DEVICE)

def extract_features_v3(ts_window):
    num_nodes = ts_window.shape[0]
    features = np.zeros((num_nodes, 15))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(num_nodes):
            ts = ts_window[i, :]
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.max(ts)
            features[i, 3] = np.min(ts)
            features[i, 4] = np.sum(ts > 0) / len(ts) if len(ts) > 0 else 0
            features[i, 5] = np.sum(ts) / len(ts) if len(ts) > 0 else 0
            if len(ts) > 5: features[i, 6] = np.mean(ts[-5:]) - np.mean(ts[:5])
            if len(ts) > 1: features[i, 7] = np.mean(np.abs(np.diff(ts)))
            features[i, 8] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 9] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 10] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            if len(ts) > 1:
                mean_val = np.mean(ts)
                if mean_val > 1e-6: features[i, 11] = np.std(ts) / mean_val
            features[i, 12] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0: features[i, 13] = (max_val - np.min(ts)) / max_val
    return np.nan_to_num(features)

def get_p_at_k(y_true, y_pred, k=5):
    idx_true = np.argsort(y_true)[-k:]
    idx_pred = np.argsort(y_pred)[-k:]
    common = len(set(idx_true) & set(idx_pred))
    return common / k

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    print(f"🚀 Validação Híbrida: ST-GCN ({WEIGHT_STGCN*100}%) + Ranking ({WEIGHT_RANKING*100}%)")
    print(f"📂 Diretório Raiz: {ROOT}")
    
    # 1. Dados
    if not os.path.exists(DATA_PATH):
        print(f"❌ Arquivo de dados não encontrado em: {DATA_PATH}")
        return

    with open(DATA_PATH, 'rb') as f: data = pickle.load(f)
    node_features = data['node_features']
    dates = data['dates']
    nodes_gdf = data.get('nodes_gdf')
    
    # Filtro Capital/RMF
    target_indices = []
    if nodes_gdf is not None:
        for idx, row in nodes_gdf.iterrows():
            reg = str(row.get('regiao', '')).lower()
            if any(x in reg for x in ['fortaleza', 'rmf']) or row.get('node_type') == 'bairro':
                target_indices.append(idx)
    print(f"🎯 Foco: {len(target_indices)} bairros (Capital/RMF)")

    # 2. Scaler
    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        print("✅ Scaler carregado")
    else:
        print("⚠️ Scaler não encontrado (Ranking pode falhar)")
    
    # 3. ST-GCN
    adj_geo = data['adj_geo']
    adj_faction = data.get('adj_faction', adj_geo)
    adj_list = [compute_norm_adj(adj_geo), compute_norm_adj(adj_faction)]
    
    stgcn = STGCN(num_nodes=319, in_channels=26, time_steps=30, num_graphs=2).to(DEVICE)
    try:
        sd = torch.load(STGCN_PATH, map_location=DEVICE)
        new_sd = {}
        for k,v in sd.items():
            nk = k.replace('module.', '')
            if nk.endswith('.gcn.weight'):
                base = nk[:-len('.gcn.weight')]
                new_sd[f"{base}.gcn.weights.0"] = v
                new_sd[f"{base}.gcn.weights.1"] = v
            else: new_sd[nk] = v
        stgcn.load_state_dict(new_sd, strict=False)
        stgcn.eval()
        print("✅ ST-GCN carregado")
    except Exception as e:
        print(f"❌ Erro ST-GCN: {e}")
        return

    # 4. Loop de Teste
    print("-" * 75)
    print(f"{'Semana Alvo':<12} | {'P@5':<6} | {'Top Real'}")
    print("-" * 75)
    
    metrics = []
    total_days = len(dates)
    
    # Testa últimas 8 semanas
    for i in range(8):
        end_idx = total_days - 7 - (i * 7)
        start_idx = end_idx - 30
        if start_idx < 0: break
        
        # ST-GCN Prediction
        X_slice = node_features[:, start_idx:end_idx, :]
        X_tensor = torch.FloatTensor(X_slice).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            stgcn_out = stgcn(X_tensor, adj_list).squeeze(0).cpu().numpy()[:, 0]
            
        # Ranking Prediction
        target_date = dates[end_idx]
        dow = target_date.weekday()
        rank_model_path = os.path.join(RANKING_DIR, f"ranking_model_day{dow}_selected.pth")
        
        ranking_score = np.zeros_like(stgcn_out)
        if os.path.exists(rank_model_path) and scaler:
            try:
                model_rank = RankingModelV3(input_dim=15).to(DEVICE)
                model_rank.load_state_dict(torch.load(rank_model_path, map_location=DEVICE))
                model_rank.eval()
                
                feats = extract_features_v3(X_slice[:, :, 0])
                feats_scaled = scaler.transform(feats)
                feats_t = torch.FloatTensor(feats_scaled).to(DEVICE)
                
                with torch.no_grad():
                    ranking_score = model_rank(feats_t).cpu().numpy()
            except: pass
            
        # Normalização Min-Max
        def normalize(v):
            if v.max() == v.min(): return np.zeros_like(v)
            return (v - v.min()) / (v.max() - v.min() + 1e-6)
            
        s_norm = normalize(stgcn_out)
        r_norm = normalize(ranking_score)
        
        # FUSÃO 90/10
        final_score = (s_norm * WEIGHT_STGCN) + (r_norm * WEIGHT_RANKING)
        
        # Ground Truth
        future = node_features[:, end_idx:end_idx+7, 0]
        y_true = np.sum(future, axis=1)
        
        # Filtro Capital
        y_true_cap = y_true[target_indices]
        y_pred_cap = final_score[target_indices]
        
        if np.sum(y_true_cap) == 0: continue
        
        p5 = get_p_at_k(y_true_cap, y_pred_cap, k=5)
        metrics.append(p5)
        
        top_node = nodes_gdf.iloc[target_indices[np.argmax(y_true_cap)]]['name']
        print(f"{str(target_date.date()):<12} | {p5:.2f}   | {top_node}")

    print("-" * 75)
    print(f"MÉDIA FINAL (Híbrido 90/10): {np.mean(metrics)*100:.1f}%")

if __name__ == "__main__":
    main()
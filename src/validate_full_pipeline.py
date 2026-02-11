import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import ndcg_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import STGCN
from src.ranking_inference import RankingInference

# CONFIGURAÇÃO
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
MODEL_PATH = os.path.join(ROOT, 'models', 'stgcn_model_v2.pth')
SCALER_PATH = os.path.join(ROOT, 'models', 'ranking_by_day', 'ranking_scaler.pkl') # NOVO
# ATIVAR RANKING!
USE_RANKING = True 
TARGET_REGION = ['fortaleza', 'rmf'] 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_norm_adj(adj):
    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt).to(DEVICE)

def get_p_at_k(y_true, y_pred, k=5):
    idx_true = np.argsort(y_true)[-k:]
    idx_pred = np.argsort(y_pred)[-k:]
    common = len(set(idx_true) & set(idx_pred))
    return common / k

def get_ndcg(y_true, y_pred, k=10):
    return ndcg_score([y_true], [y_pred], k=k)

# Mesma extração do Treino
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
            if len(ts) > 5:
                features[i, 6] = np.mean(ts[-5:]) - np.mean(ts[:5])
            if len(ts) > 1:
                features[i, 7] = np.mean(np.abs(np.diff(ts)))
            features[i, 8] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 9] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 10] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            if len(ts) > 1:
                features[i, 11] = np.mean(np.abs(np.diff(ts)))
                mean_val = np.mean(ts)
                if mean_val > 1e-6: features[i, 12] = np.std(ts) / mean_val
            features[i, 13] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0: features[i, 14] = (max_val - np.min(ts)) / max_val
    return np.nan_to_num(features)

class ModelValidator:
    def __init__(self):
        self.load_data()
        self.load_models()
        
    def load_data(self):
        print(f"📦 Carregando dados...")
        with open(DATA_PATH, 'rb') as f: data = pickle.load(f)
        self.node_features = data['node_features'] 
        self.dates = data['dates']
        self.nodes_gdf = data.get('nodes_gdf')
        
        # Carregar Scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler carregado com sucesso!")
        else:
            print("⚠️ Scaler NÃO encontrado. O Ranking vai falhar.")
            self.scaler = None

        self.target_indices = []
        if self.nodes_gdf is not None:
            for idx, row in self.nodes_gdf.iterrows():
                reg = str(row.get('regiao', '')).lower()
                if any(x in reg for x in TARGET_REGION) or row.get('node_type') == 'bairro':
                    self.target_indices.append(idx)
        
        adj_geo = data['adj_geo']
        adj_faction = data.get('adj_faction', adj_geo)
        self.adj_list = [compute_norm_adj(adj_geo), compute_norm_adj(adj_faction)]

    def load_models(self):
        print(f"🤖 Carregando Modelos...")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k.replace('module.', '')
            if nk.endswith('.gcn.weight'):
                base = nk[:-len('.gcn.weight')]
                new_state_dict[f"{base}.gcn.weights.0"] = v
                new_state_dict[f"{base}.gcn.weights.1"] = v
            else:
                new_state_dict[nk] = v
        self.model = STGCN(num_nodes=319, in_channels=26, time_steps=30, num_graphs=2).to(DEVICE)
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        # Carrega Ranking manual para ter controle total (bypass na classe antiga)
        self.ranking_net = None
        ranking_path = os.path.join(ROOT, 'models', 'ranking_by_day', 'ranking_model_day2_selected.pth')
        if USE_RANKING and os.path.exists(ranking_path):
            # Definição local da classe para garantir compatibilidade
            class RankingModelV3(torch.nn.Module):
                def __init__(self, input_dim=15, hidden_dim=128):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(hidden_dim, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 1)
                    )
                def forward(self, x): return self.net(x)
            
            self.ranking_net = RankingModelV3().to(DEVICE)
            self.ranking_net.load_state_dict(torch.load(ranking_path, map_location=DEVICE))
            self.ranking_net.eval()
            print("⚖️  Ranking V3 Carregado Manualmente (Bypassing Inference Class)")

    def run_backtest(self, test_weeks=8):
        print(f"\n🚀 Backtest Final: ST-GCN + Ranking V3 (Normalizado)")
        print("="*100)
        print(f"{'Semana Alvo':<12} | {'P@5':<6} | {'P@10':<6} | {'Top Real'}")
        print("-" * 100)
        
        metrics = {'p5': [], 'p10': [], 'ndcg': []}
        total_days = len(self.dates)
        window = 30
        horizon = 7
        
        for w in range(test_weeks):
            end_idx = total_days - horizon - (w * 7)
            start_idx = end_idx - window
            if start_idx < 0: break
            
            X_slice = self.node_features[:, start_idx:end_idx, :]
            X_tensor = torch.FloatTensor(X_slice).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            future_slice = self.node_features[:, end_idx:end_idx+horizon, 0]
            y_true_full = np.sum(future_slice, axis=1)
            
            if np.sum(y_true_full) == 0: continue

            # 1. ST-GCN Raw Score
            with torch.no_grad():
                pred = self.model(X_tensor, self.adj_list)
                stgcn_score = pred.squeeze(0).cpu().numpy()[:, 0]
            
            # 2. Ranking Correction (Agora com Scaler!)
            final_score = stgcn_score
            if self.ranking_net and self.scaler:
                # Extrair features
                feats = extract_features_v3(X_slice[:, :, 0])
                # NORMALIZAR! (O segredo)
                feats_scaled = self.scaler.transform(feats)
                feats_t = torch.FloatTensor(feats_scaled).to(DEVICE)
                
                with torch.no_grad():
                    rank_score = self.ranking_net(feats_t).cpu().numpy()[:, 0]
                
                # Combinação: ST-GCN (Espacial) * Ranking (Temporal)
                # Normaliza ambos para 0-1 antes de multiplicar
                stgcn_norm = (stgcn_score - stgcn_score.min()) / (stgcn_score.max() - stgcn_score.min() + 1e-6)
                rank_norm = (rank_score - rank_score.min()) / (rank_score.max() - rank_score.min() + 1e-6)
                
                final_score = (stgcn_norm * 0.6) + (rank_norm * 0.4)

            # Filtro Capital
            y_true = y_true_full[self.target_indices]
            y_pred = final_score[self.target_indices]
            nodes = self.nodes_gdf.iloc[self.target_indices]

            p5 = get_p_at_k(y_true, y_pred, k=5)
            p10 = get_p_at_k(y_true, y_pred, k=10)
            
            metrics['p5'].append(p5)
            metrics['p10'].append(p10)
            
            top_name = nodes.iloc[np.argmax(y_true)]['name']
            date_target = self.dates[end_idx].date()
            print(f"{str(date_target):<12} | {p5:.2f}   | {p10:.2f}   | {top_name}")

        print("=" * 100)
        print(f"MÉDIA FINAL (COM RANKING CORRIGIDO):")
        print(f"Precision@5:  {np.mean(metrics['p5'])*100:.1f}%")
        print("=" * 100)

if __name__ == "__main__":
    validator = ModelValidator()
    validator.run_backtest()
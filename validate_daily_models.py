import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

# Configuração
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
MODELS_DIR = os.path.join(ROOT, 'models', 'ranking_by_day')
SCALER_PATH = os.path.join(MODELS_DIR, 'ranking_scaler.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. ARQUITETURA V3 (15 Inputs - Igual ao Treino)
# ==============================================================================
class RankingModelDay(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=128):
        super(RankingModelDay, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

# ==============================================================================
# 2. Extração de Features V3 (Estatísticas Robustas)
# ==============================================================================
def extract_features_v3(ts_window):
    """Extrai as 15 features estatísticas usadas no treino V3"""
    num_nodes = ts_window.shape[0]
    features = np.zeros((num_nodes, 15))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(num_nodes):
            ts = ts_window[i, :]
            # Básicas
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.max(ts)
            features[i, 3] = np.min(ts)
            features[i, 4] = np.sum(ts > 0) / len(ts) if len(ts) > 0 else 0
            features[i, 5] = np.sum(ts) / len(ts) if len(ts) > 0 else 0
            
            # Tendência
            if len(ts) > 5:
                features[i, 6] = np.mean(ts[-5:]) - np.mean(ts[:5])
            if len(ts) > 1:
                features[i, 7] = np.mean(np.abs(np.diff(ts)))
            
            # Recência
            features[i, 8] = np.mean(ts[-3:]) if len(ts) >= 3 else 0
            features[i, 9] = np.mean(ts[-7:]) if len(ts) >= 7 else 0
            features[i, 10] = np.mean(ts[-14:]) if len(ts) >= 14 else 0
            
            # Volatilidade
            if len(ts) > 1:
                features[i, 11] = np.mean(np.abs(np.diff(ts)))
                mean_val = np.mean(ts)
                if mean_val > 1e-6:
                    features[i, 12] = np.std(ts) / mean_val
            
            # Picos
            features[i, 13] = np.percentile(ts, 75) - np.percentile(ts, 25)
            max_val = np.max(ts)
            if max_val > 0: 
                features[i, 14] = (max_val - np.min(ts)) / max_val

    return np.nan_to_num(features)

def get_p_at_k(y_true, y_pred, k=5):
    idx_true = np.argsort(y_true)[-k:]
    idx_pred = np.argsort(y_pred)[-k:]
    common = len(set(idx_true) & set(idx_pred))
    return common / k

def evaluate_day(day_idx, X_test, y_test, model_path, scaler):
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo não encontrado para Dia {day_idx}")
        return None

    # Carregar modelo V3
    model = RankingModelDay(input_dim=15).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"❌ Erro arquitetura Dia {day_idx}: {e}")
        return None

    # Extrair Features V3 (15 dims)
    X_feats = extract_features_v3(X_test)
    
    # Normalizar com Scaler (CRÍTICO)
    if scaler:
        X_scaled = scaler.transform(X_feats)
    else:
        X_scaled = X_feats # Fallback perigoso
        
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    p5 = get_p_at_k(y_test, y_pred, k=5)
    
    try:
        ndcg = ndcg_score([y_test], [y_pred], k=10)
    except:
        ndcg = 0.0

    return {'p5': p5, 'ndcg': ndcg}

# ==============================================================================
# 3. Loop de Validação
# ==============================================================================
def main():
    print(f"🚀 Iniciando Validação V3 por Dia da Semana...")
    
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    node_features = data['node_features'][:, :, 0] # Apenas CVLI
    dates = pd.to_datetime(data['dates'])
    
    # Carregar Scaler
    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler carregado.")
    else:
        print("⚠️ AVISO: Scaler não encontrado! Resultados serão ruins.")

    results = []
    
    # Testar a última semana completa disponível nos dados
    test_window_start = len(dates) - 37 
    
    print("-" * 60)
    print(f"{'Dia':<10} | {'Data':<12} | {'Modelo':<25} | {'P@5':<6} | {'NDCG':<6}")
    print("-" * 60)

    for i in range(7):
        target_idx = test_window_start + 30 + i
        if target_idx >= len(dates): break
        
        target_date = dates[target_idx]
        dow = target_date.weekday()
        
        model_name = f"ranking_model_day{dow}_selected.pth"
        model_path = os.path.join(MODELS_DIR, model_name)

        # Dados: 30 dias passados
        X_window = node_features[:, target_idx-30:target_idx]
        y_test_day = node_features[:, target_idx]

        metrics = evaluate_day(dow, X_window, y_test_day, model_path, scaler)
        
        if metrics:
            day_name = target_date.strftime("%A")
            date_str = target_date.strftime("%Y-%m-%d")
            print(f"{day_name:<10} | {date_str:<12} | {model_name:<25} | {metrics['p5']:.2f}   | {metrics['ndcg']:.2f}")
            results.append(metrics)

    if results:
        avg_p5 = np.mean([r['p5'] for r in results])
        avg_ndcg = np.mean([r['ndcg'] for r in results])
        print("-" * 60)
        print(f"MÉDIA SEMANAL V3: P@5 = {avg_p5:.4f} ({avg_p5*100:.1f}%) | NDCG = {avg_ndcg:.4f}")
        print("✅ Validação concluída.")
    else:
        print("❌ Falha na validação.")

if __name__ == "__main__":
    main()
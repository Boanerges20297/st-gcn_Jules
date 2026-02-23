import pickle
import numpy as np
import torch
import os
import sys
import pandas as pd
import json

# Adicionar raiz ao path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.architectures import DeepSTGAT_64

def validate_cascade():
    # 1. Carregar Configuracoes e Dados
    config_path = os.path.join(ROOT, 'outputs', 'top40_static_2024_2025.json')
    data_path = os.path.join(ROOT, 'data', 'processed', 'processed_fortaleza.pkl')
    model_path = os.path.join(ROOT, 'models', 'active', 'fortaleza_model.pth')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    top40_indices = config['top40_indices']
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['node_features']
    dates = pd.to_datetime(data['dates'])
    adj_geo = data['adj_geo']
    adj_conf = data['adj_conflict']
    
    # 2. Carregar Modelo Generalista (Campeao)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSTGAT_64(num_nodes=features.shape[0], in_channels=29, time_steps=30).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 3. Preparar Ground Truth (Ultimos 90 dias - REALITY)
    test_days = 90
    test_indices = range(len(dates) - test_days, len(dates))
    
    # Normalizacao (mesma do treino)
    features_norm = features.copy()
    for c in range(29):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features[:, :, c] - m) / s

    def n(a):
        s = np.array(a.sum(1)); d = np.power(s, -0.5).flatten(); d[np.isinf(d)]=0.; m=np.diag(d)
        return torch.from_numpy(a.dot(m).transpose().dot(m)).float().to(device)
    adj_list = [n(adj_geo), n(adj_conf)]

    print(f"--- VALIDACAO EM CASCATA (PENEIRA ESTATICA TOP 40) ---")
    print(f"Periodo: {dates[test_indices[0]].date()} ate {dates[test_indices[-1]].date()}")
    
    p20_scores = []
    
    for t in test_indices:
        # Previsao Neural (Global)
        x_t = torch.tensor(features_norm[:, t-30:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_global = model(x_t, adj_list).squeeze().cpu().numpy()
        
        # Realidade (Proximos 7 dias)
        # Note: No treino original, o target e a soma dos 7 dias a frente.
        # Para validadacao honesta, olhamos a janela de 7 dias a partir de t.
        if t + 7 >= len(dates): continue
        y_true = features[:, t:t+7, 0].sum(axis=1)
        
        if y_true.sum() == 0: continue
        
        # --- APLICACAO DA CASCATA ---
        # 1. Pegamos apenas os scores neurais dos 40 bairros da peneira estatica
        cascade_scores = np.zeros_like(pred_global) - 999999 # Mata quem nao esta na peneira
        for idx in top40_indices:
            cascade_scores[idx] = pred_global[idx]
            
        # 2. Geramos o Top 20 Final a partir dessa peneira
        top_20_pred = np.argsort(-cascade_scores)[:20]
        top_20_true = np.argsort(-y_true)[:20]
        
        hits = len(set(top_20_pred) & set(top_20_true))
        p20_scores.append(hits / 20.0)

    avg_p20 = np.mean(p20_scores) if p20_scores else 0
    print(f"\n✅ RESULTADO CASCATA: {avg_p20*100:.2f}%")
    print(f"📊 Comparativo: Modelo Generalista Puro era 63.20%")
    print("="*50)

if __name__ == "__main__":
    validate_cascade()

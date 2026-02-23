import pickle
import numpy as np
import torch
import os
import sys
import pandas as pd

# Adicionar raiz ao path (subindo dois niveis a partir de scripts/diversos)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.architectures import DeepSTGAT_64

def diagnose_frio():
    # Caminho absoluto para evitar erro no Windows
    model_path = os.path.join(ROOT, 'models', 'test', 'ranking', 'fortaleza_expert_frio.pth')
    data_path = os.path.join(ROOT, 'data', 'processed', 'processed_fortaleza.pkl')
    
    print(f"Buscando modelo em: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Modelo nao encontrado. Arquivos no diretorio: {os.listdir(os.path.dirname(model_path)) if os.path.exists(os.path.dirname(model_path)) else 'Dir nao existe'}")
        return

    # Carregar Dados
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['node_features']
    nodes_gdf = data['nodes_gdf']
    dates = pd.to_datetime(data['dates'])
    
    # Carregar Modelo (Ajustado para PyTorch 2.6+)
    model = DeepSTGAT_64(num_nodes=len(nodes_gdf), in_channels=29, time_steps=30)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Pegar apenas meses FRIOS do histórico recente para teste
    df_monthly = pd.DataFrame({'date': dates, 'cvli': features[:, :, 0].sum(axis=0)})
    df_monthly['year_month'] = df_monthly['date'].dt.to_period('M')
    monthly_sums = df_monthly.groupby('year_month')['cvli'].sum()
    frio_periods = monthly_sums[monthly_sums < 60].index
    
    test_indices = []
    for p in frio_periods:
        indices = np.where(df_monthly['year_month'] == p)[0]
        test_indices.extend([i for i in indices if i >= 30 and i < len(dates) - 7])
    
    # Analisar Erros
    errors_list = []
    
    # Normalizacao (mesma do treino)
    features_norm = features.copy()
    for c in range(29):
        m, s = features[:, :, c].mean(), features[:, :, c].std() + 1e-6
        features_norm[:, :, c] = (features[:, :, c] - m) / s

    adj_geo_t = torch.tensor(data['adj_geo'], dtype=torch.float32) # Simplificado para CPU
    adj_conf_t = torch.tensor(data['adj_conflict'], dtype=torch.float32)

    print(f"--- DIAGNOSTICO ERROS ESPECIALISTA FRIO ({len(test_indices)} dias analisados) ---")
    
    for t in test_indices[-20:]: # Analisar os ultimos 20 dias frios
        x_t = torch.tensor(features_norm[:, t-30:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y_true = features[:, t+1:t+8, 0].sum(axis=1) # Proximos 7 dias
        
        if y_true.sum() == 0: continue
            
        with torch.no_grad():
            # Simulacao do GAT em CPU (matrizes identidade para simplificar diagnostico)
            pred = model(x_t, [torch.eye(len(nodes_gdf)), torch.eye(len(nodes_gdf))]).squeeze().numpy()
        
        top_10_true = np.argsort(-y_true)[:10]
        top_40_pred = np.argsort(-pred)[:40]
        
        missed = [idx for idx in top_10_true if idx not in top_40_pred and y_true[idx] > 0]
        
        for m_idx in missed:
            errors_list.append({
                'date': str(dates[t].date()),
                'bairro': nodes_gdf.iloc[m_idx]['name'],
                'facao': nodes_gdf.iloc[m_idx]['faction'],
                'crimes_reais': y_true[m_idx]
            })

    if not errors_list:
        print("✅ Nao foram encontrados erros gritantes nos ultimos 20 dias analisados.")
    else:
        df_errors = pd.DataFrame(errors_list)
        print("\n📍 BAIRROS CRITICOS QUE O MODELO ESTA PERDENDO (Missed @ 40):")
        summary = df_errors.groupby('bairro').agg({'crimes_reais': 'sum', 'facao': 'first'}).sort_values(by='crimes_reais', ascending=False)
        print(summary.head(15))
        
        print("\n🚩 FACCOES MAIS 'INVISIVEIS' PARA O MODELO:")
        print(df_errors['facao'].value_counts())

if __name__ == "__main__":
    diagnose_frio()

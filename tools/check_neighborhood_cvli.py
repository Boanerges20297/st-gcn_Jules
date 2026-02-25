import pickle
import numpy as np
import os
import pandas as pd

def check_cvli_stats():
    path = 'data/processed/processed_fortaleza.pkl'
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        return
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['node_features'] # (N, T, C)
    nodes_gdf = data['nodes_gdf']
    
    # Analisar últimos 120 dias
    WINDOW = 120
    cvli_recent = features[:, -WINDOW:, 0]
    months_recent = WINDOW / 30.0
    
    node_stats = []
    for i in range(cvli_recent.shape[0]):
        total_cvli = cvli_recent[i].sum()
        avg_per_month = total_cvli / months_recent
        node_stats.append({
            'name': nodes_gdf.iloc[i]['name'],
            'total_cvli_120d': total_cvli,
            'avg_per_month_120d': avg_per_month
        })
    
    df = pd.DataFrame(node_stats)
    df = df.sort_values(by='avg_per_month_120d', ascending=False)
    
    print(f"Total de bairros: {len(df)}")
    print(f"Bairros >= 3 CVLI/mês (nos últimos 120 dias): {len(df[df['avg_per_month_120d'] >= 3])}")
    print(f"Bairros >= 1 CVLI/mês (nos últimos 120 dias): {len(df[df['avg_per_month_120d'] >= 1])}")
    print(f"Bairros >= 0.75 CVLI/mês (nos últimos 120 dias): {len(df[df['avg_per_month_120d'] >= 0.75])}")
    
    print("\nTop 10 Bairros (Últimos 120 dias):")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    check_cvli_stats()

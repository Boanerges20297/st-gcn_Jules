import pickle
import numpy as np
import pandas as pd
import os
import sys

# Adicionar raiz ao path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def generate_top40_analysis():
    path = os.path.join(ROOT, 'data', 'processed', 'processed_fortaleza.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['node_features'] # (Nodes, Days, Channels)
    dates = pd.to_datetime(data['dates'])
    nodes_gdf = data['nodes_gdf']
    
    # 1. Filtrar apenas 2025
    mask_2025 = (dates >= '2025-01-01') & (dates <= '2025-12-31')
    data_2025 = features[:, mask_2025, 0]
    dates_2025 = dates[mask_2025]
    
    print(f"--- ANALISE BRUTA CVLI 2025 ---")
    print(f"Periodo: {dates_2025.min().date()} ate {dates_2025.max().date()}")
    
    # 2. Calcular Soma Bruta por Bairro
    cvli_sum = data_2025.sum(axis=1)
    
    # Criar DataFrame para facil visualizacao
    df_ranking = pd.DataFrame({
        'node_idx': range(len(nodes_gdf)),
        'bairro': nodes_gdf['name'].values,
        'cvli_total': cvli_sum
    })
    
    # 3. Gerar Top 40
    top_40 = df_ranking.nlargest(40, 'cvli_total')
    
    print("\n🏆 TOP 10 BAIRROS MAIS CRITICOS DE 2025:")
    print(top_40.head(10).to_string(index=False))
    
    # 4. Separar 90 dias para teste
    test_days = 90
    train_data = data_2025[:, :-test_days]
    test_data = data_2025[:, -test_days:]
    
    print(f"\n--- DIVISAO DE DADOS (2025) ---")
    print(f"Treino: {train_data.shape[1]} dias")
    print(f"Teste (Lastro): {test_data.shape[1]} dias ({dates_2025[-test_days].date()} ate {dates_2025[-1].date()})")
    
    # Salvar a lista de indices do Top 40 para o Micro-Especialista
    top40_indices = top_40['node_idx'].tolist()
    
    output_info = {
        'top40_indices': top40_indices,
        'top40_names': top_40['bairro'].tolist(),
        'test_start_date': str(dates_2025[-test_days].date())
    }
    
    # Salvar metadados para a proxima fase
    import json
    with open(os.path.join(ROOT, 'outputs', 'top40_2025_config.json'), 'w', encoding='utf-8') as f:
        json.dump(output_info, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Lista Top 40 e configuracao salvas em: outputs/top40_2025_config.json")

if __name__ == "__main__":
    generate_top40_analysis()

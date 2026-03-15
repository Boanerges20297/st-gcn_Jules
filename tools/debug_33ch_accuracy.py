import torch
import numpy as np
import pandas as pd
from src.core.orchestrator import StateOrchestrator
import os

def test_model_accuracy():
    print("🔍 Iniciando Auditoria de Emergência - Modelo 33 Canais")
    orch = StateOrchestrator(os.getcwd())
    
    if 'fortaleza' not in orch.specialists:
        print("❌ Especialista Fortaleza não carregado.")
        return

    spec = orch.specialists['fortaleza']
    model = spec['model']
    data = spec['data']
    
    print(f"✅ Modelo: {spec['channels']} canais carregados.")
    
    # Simular a predição do Orquestrador (que calcula o Momentum dinamicamente)
    results = orch.get_combined_risk()
    
    # Pegar o Ground Truth (CVLI real do último dia disponível nos dados)
    node_features = data['node_features']
    last_day_cvli = node_features[:, -1, 0] # Canal 0 é CVLI
    
    # Se o último dia for tudo zero (esperado se os dados não foram atualizados hoje), 
    # vamos buscar o último dia que teve crime para validar o ranking.
    day_idx = -1
    while np.sum(node_features[:, day_idx, 0]) == 0 and day_idx > -30:
        day_idx -= 1
    
    actual_crimes = node_features[:, day_idx, 0]
    num_crimes = np.sum(actual_crimes > 0)
    
    print(f"📅 Validando contra o dia (index {day_idx}) que teve {num_crimes} crimes.")
    
    # Rodar predição para o dia anterior a esse crime
    # (O orquestrador já faz isso, mas vamos garantir o ranking)
    sorted_nodes = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_10_names = [name for name, score in sorted_nodes[:10]]
    
    # Mapear nomes de volta para índices
    from src.core.orchestrator import normalize_name
    name_to_idx = {normalize_name(row['name']): i for i, row in data['nodes_gdf'].iterrows()}
    
    # Teste em Janela Deslizante (Últimos 7 dias de dados reais)
    print("\n--- TESTE DE ESTRESSE (Últimos 7 dias) ---")
    history_p10 = []
    
    for d in range(-1, -8, -1):
        actual_crimes = node_features[:, d, 0]
        num_crimes = np.sum(actual_crimes > 0)
        
        # Simular predição para o estado D-1
        # Para simplificar o debug, vamos usar o get_combined_risk 
        # (Idealmente teríamos que injetar o estado exato do dia d-1)
        # mas como get_combined_risk usa o final dos dados, vamos testar apenas o P@10 final
        
        top_10_indices = [name_to_idx[name] for name in top_10_names if name in name_to_idx]
        hits = np.sum(actual_crimes[top_10_indices] > 0)
        p_at_10 = (hits / 10) * 100
        history_p10.append(p_at_10)
        
        print(f"Dia {d}: Crimes={num_crimes} | Hits no Top 10={hits} | P@10={p_at_10:.1f}%")
    
    print(f"\n📊 Média Final P@10: {np.mean(history_p10):.2f}%")

if __name__ == "__main__":
    test_model_accuracy()

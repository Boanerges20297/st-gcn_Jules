import torch
import pickle
import numpy as np
import os
import sys

# Adiciona Phase4 ao path para importar o modelo
sys.path.append(os.path.join(os.getcwd(), 'Phase4'))
from model_v4 import DeepSTGAT

def test_sensitivity():
    print("--- INICIANDO TESTE DE SENSIBILIDADE TATICA (PHASE 5) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('data/processed/processed_graph_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    adj_list = [torch.from_numpy(data['adj_geo']).float().to(device), 
                torch.from_numpy(data['adj_conflict']).float().to(device)]
    
    num_nodes = data['node_features'].shape[0]
    in_channels = 27
    history_window = 30 
    
    model = DeepSTGAT(num_nodes=num_nodes, in_channels=in_channels, time_steps=history_window).to(device)
    
    model_path = 'models/phase5/best_stgat_v5_massive.pth'
    if not os.path.exists(model_path):
        print(f"Erro: Modelo nao encontrado em {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo carregado com P@10 recorde: {checkpoint.get('p10', 0):.4f}")

    base_input = torch.zeros((1, in_channels, num_nodes, history_window)).to(device)
    
    with torch.no_grad():
        base_out = model(base_input, adj_list).squeeze().cpu().numpy()

    target_node = 100
    bairro_name = data['nodes_gdf'].iloc[target_node]['name']
    print(f"\nTestando Bairro Alvo: {bairro_name} (ID {target_node})")

    scenarios = [
        ("Base (Silencio)", None, 0),
        ("Roubo de Veiculo (C1)", 1, 1.0),
        ("Choque de Inteligencia (C25)", 25, 1.0),
        ("Expulsao de Morador (C25)", 25, 2.0),
        ("Alerta de Incursao (C26)", 26, 1.0)
    ]

    results = []
    for name, channel, val in scenarios:
        test_input = base_input.clone()
        if channel is not None:
            test_input[0, channel, target_node, -3:] = val
        
        with torch.no_grad():
            out = model(test_input, adj_list).squeeze().cpu().numpy()
            risk = out[target_node]
            results.append((name, risk))

    print("\nResultados de Sensibilidade (Score Bruto):")
    base_risk = results[0][1]
    for name, risk in results:
        diff = risk - base_risk
        impact = (diff / abs(base_risk)) * 100 if base_risk != 0 else 0
        print(f"  {name:30} | Score: {risk:8.4f} | Impacto: {impact:+7.2f}%")

if __name__ == "__main__":
    test_sensitivity()

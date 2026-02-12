import pickle
import torch
import numpy as np
import os
import sys

# Adicionar o diretório atual ao path para importar model_v4
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_v4 import DeepSTGAT

DATA_FILE = 'data/processed/processed_graph_data.pkl'

def test():
    print("Iniciando teste de setup...")
    
    if not os.path.exists(DATA_FILE):
        print(f"ERRO: Arquivo {DATA_FILE} não encontrado!")
        return

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    node_features = data['node_features']
    print(f"Node Features Shape: {node_features.shape}") # Esperado: (N, T, C)
    
    N, T, C = node_features.shape
    HISTORY_WINDOW = 30
    
    # Teste de inicialização do modelo
    print("Inicializando modelo...")
    try:
        model = DeepSTGAT(num_nodes=N, in_channels=C, time_steps=HISTORY_WINDOW)
        print("Modelo inicializado com sucesso!")
    except Exception as e:
        print(f"ERRO ao inicializar modelo: {e}")
        return

    # Teste de um forward pass com dados dummy
    print("Testando Forward Pass com dados dummy...")
    try:
        dummy_x = torch.randn(2, C, N, HISTORY_WINDOW)
        adj_list = [torch.eye(N), torch.eye(N)]
        output = model(dummy_x, adj_list)
        print(f"Forward Pass bem sucedido! Output shape: {output.shape}")
    except Exception as e:
        print(f"ERRO no Forward Pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()

"""
Script para criar um modelo STGCN stub/mínimo para testes da API.
Carrega os dados processados e treina/salva um checkpoint mínimo.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from model import STGCN
from torch.utils.data import DataLoader, TensorDataset

import sys
import os as os_module
sys.path.insert(0, os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))

DATA_FILE = os_module.path.join(os_module.path.dirname(__file__), '..', 'data', 'processed', 'processed_graph_data.pkl')
MODEL_PATH = os_module.path.join(os_module.path.dirname(__file__), '..', 'models', 'stgcn_model.pth')

def train_stub():
    """Treina modelo por 1-2 épocas para criar checkpoint válido."""
    print("Carregando dados...")
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    node_features = data_pack['node_features']  # (N, T, F)
    adj_matrix = data_pack['adj_matrix']        # (N, N)

    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]
    history_window = 7

    print(f"Dados carregados:")
    print(f"  - Nós: {num_nodes}")
    print(f"  - Timesteps: {node_features.shape[1]}")
    print(f"  - Features: {num_features}")

    # Criar modelo
    print("Criando modelo STGCN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")

    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=history_window)
    model.to(device)
    model.eval()

    print(f"Modelo STGCN criado com sucesso")
    print(f"  - num_nodes: {num_nodes}")
    print(f"  - in_channels: {num_features}")
    print(f"  - time_steps: {history_window}")

    # Salvar checkpoint
    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✓ Modelo salvo em {MODEL_PATH}")

if __name__ == "__main__":
    train_stub()

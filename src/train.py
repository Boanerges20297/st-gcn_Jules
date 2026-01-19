import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from src.model import STGCN
from torch.utils.data import DataLoader, TensorDataset

DATA_FILE = 'data/processed_graph_data.pkl'
MODEL_PATH = 'models/stgcn_model.pth'
HISTORY_WINDOW = 7
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001

LOSS_WEIGHTS = torch.tensor([5.0, 1.0]) 

def prepare_dataset(node_features):
    num_nodes, num_timesteps, num_features = node_features.shape
    X, Y = [], []
    for i in range(num_timesteps - HISTORY_WINDOW):
        window = node_features[:, i:i+HISTORY_WINDOW, :]
        target = node_features[:, i+HISTORY_WINDOW, :]
        X.append(np.transpose(window, (2, 0, 1)))
        Y.append(target)
    return np.array(X), np.array(Y)

def weighted_mse_loss(input, target, weights):
    sq_diff = (input - target) ** 2
    weighted_sq_diff = sq_diff * weights.view(1, 1, -1)
    return torch.mean(weighted_sq_diff)

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    print("Carregando dados...")
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)
        
    node_features = data_pack['node_features']
    adj_matrix = data_pack['adj_matrix']
    
    adj_tensor = torch.FloatTensor(adj_matrix)
    rowsum = adj_tensor.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt)
    
    print("Criando janelas temporais...")
    X, Y = prepare_dataset(node_features)
    
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    print(f"Treino: {X_train.shape}, Validação: {X_val.shape}")
    
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]
    
    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=HISTORY_WINDOW).to(device)
    norm_adj = norm_adj.to(device)
    loss_weights = LOSS_WEIGHTS.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Iniciando treinamento...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, norm_adj)
            loss = weighted_mse_loss(output, batch_y, loss_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x, norm_adj)
                loss = weighted_mse_loss(output, batch_y, loss_weights)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from src.model import STGCN
from torch.utils.data import DataLoader, TensorDataset

DATA_FILE = 'data/processed_graph_data.pkl'
MODEL_CVLI_PATH = 'models/cvli_model.pth'
MODEL_CVP_PATH = 'models/cvp_model.pth'

# Configurations
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001

# Model specific configurations
CVLI_CONFIG = {
    'input_window': 180,
    'prediction_window': 30, # We will predict the sum/risk for the next 30 days
    'target_feature_idx': 0, # CVLI is index 0
    'model_path': MODEL_CVLI_PATH
}

CVP_CONFIG = {
    'input_window': 30,
    'prediction_window': 7, # We will predict the sum/risk for the next 7 days
    'target_feature_idx': 1, # CVP is index 1
    'model_path': MODEL_CVP_PATH
}

def prepare_dataset(node_features, config):
    input_window = config['input_window']
    prediction_window = config['prediction_window']
    feature_idx = config['target_feature_idx']

    num_nodes, num_timesteps, num_features = node_features.shape
    X, Y = [], []

    # We need enough data for input + prediction
    for i in range(num_timesteps - input_window - prediction_window):
        # Input: All features for the input window
        window = node_features[:, i:i+input_window, :]

        # Target: Sum of the specific feature for the prediction window
        target_seq = node_features[:, i+input_window : i+input_window+prediction_window, feature_idx]

        # We want to predict a single value representing the risk (e.g., total count)
        # Shape of target_seq: (Num_Nodes, Prediction_Window)
        # We sum over time to get total events per node
        target = np.sum(target_seq, axis=1) # Shape: (Num_Nodes,)

        # The model output size is num_classes=1 (regression for this specific risk)
        target = target.reshape(num_nodes, 1)

        X.append(np.transpose(window, (2, 0, 1))) # (C, N, T)
        Y.append(target)

    return np.array(X), np.array(Y)

def train_model(config, node_features, norm_adj, device):
    print(f"Training model for configuration: {config}")
    
    X, Y = prepare_dataset(node_features, config)
    
    if len(X) == 0:
        print("Not enough data points for this configuration.")
        return
    
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")
    
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    num_nodes = node_features.shape[0]
    num_features = node_features.shape[2]
    
    # Instantiate model
    # Output classes = 1 (Regression for the specific risk count)
    model = STGCN(num_nodes=num_nodes, in_channels=num_features, time_steps=config['input_window'], num_classes=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, norm_adj)
            # Output shape (B, N, 1), Batch Y shape (B, N, 1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x, norm_adj)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), config['model_path'])
    print(f"Model saved to {config['model_path']}")

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    print("Loading data...")
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)

    node_features = data_pack['node_features']
    adj_matrix = data_pack['adj_matrix']

    # Precompute normalized adjacency
    adj_tensor = torch.FloatTensor(adj_matrix)
    rowsum = adj_tensor.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tensor), d_mat_inv_sqrt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    norm_adj = norm_adj.to(device)

    # Train CVLI Model
    train_model(CVLI_CONFIG, node_features, norm_adj, device)

    # Train CVP Model
    train_model(CVP_CONFIG, node_features, norm_adj, device)

if __name__ == "__main__":
    main()

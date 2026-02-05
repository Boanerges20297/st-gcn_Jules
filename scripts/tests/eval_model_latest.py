"""
Avaliar modelo treinado vs baseline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import torch
from src.model import STGCN
from torch.utils.data import TensorDataset, DataLoader

DATA_FILE = 'data/processed/processed_graph_data.pkl'
MODEL_PATH = 'models/stgcn_model_v2.pth'
HISTORY_WINDOW = 30

def evaluate_model():
    # Load data
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)
    
    node_features = data_pack['node_features']  # (319, 1491, 8)
    adjacency = data_pack['adjacency']
    
    print(f"Data shape: node_features={node_features.shape}")
    print(f"Adjacency shape: {adjacency.shape}")
    
    # Create windows (like in train.py)
    num_nodes, num_timesteps, num_features = node_features.shape
    
    # Window creation
    X = []
    Y = []
    for t in range(num_timesteps - HISTORY_WINDOW):
        # X: past 30 days
        X.append(node_features[:, t:t+HISTORY_WINDOW, :])
        # Y: next day
        Y.append(node_features[:, t+HISTORY_WINDOW, :2])  # CVLI + CVP only
    
    X = np.array(X)  # (1461, 319, 30, 8)
    Y = np.array(Y)  # (1461, 319, 2)
    
    # Reshape for model: (batch, features, nodes, time)
    X = np.transpose(X, (0, 3, 1, 2))  # (1461, 8, 319, 30)
    
    # Load model
    device = torch.device('cpu')
    model = STGCN(num_nodes=319, num_features=8, num_timesteps=HISTORY_WINDOW, adjacency=adjacency).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[OK] Model loaded from {MODEL_PATH}")
    else:
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return
    
    model.eval()
    
    # Evaluate
    with torch.no_grad():
        predictions = model(X_val.to(device)).cpu().numpy()
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Y_val shape: {Y_val.shape}")
    
    # Calculate metrics
    # P@5: proportion of top-5% nodes with crime that model predicted as top-5%
    def calculate_p_at_5(y_true, y_pred):
        """P@5: precision at top 5%"""
        n_nodes = y_true.shape[1]
        threshold = int(n_nodes * 0.05)  # top 5%
        
        precisions = []
        for i in range(len(y_true)):
            # True positive: nodes with crime
            true_positives = np.where(y_true[i] > 0)[0]
            if len(true_positives) == 0:
                continue
            
            # Predicted top-5%
            pred_top5 = np.argsort(y_pred[i])[-threshold:]
            
            # Hits
            hits = len(np.intersect1d(true_positives, pred_top5))
            precision = hits / len(true_positives) if len(true_positives) > 0 else 0
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0
    
    p_at_5 = calculate_p_at_5(Y_val.numpy(), predictions)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"P@5 (current): {p_at_5:.4f} ({p_at_5*100:.2f}%)")
    print(f"Baseline:      0.1489 (14.89%)")
    print(f"Difference:    {(p_at_5 - 0.1489)*100:+.2f}pp")
    print(f"{'='*80}")
    
    # Stats
    print(f"\nPrediction stats:")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std:  {predictions.std():.4f}")
    print(f"  Min:  {predictions.min():.4f}")
    print(f"  Max:  {predictions.max():.4f}")
    
    print(f"\nTrue label stats:")
    print(f"  Mean: {Y_val.mean():.4f}")
    print(f"  Std:  {Y_val.std():.4f}")
    print(f"  Min:  {Y_val.min():.4f}")
    print(f"  Max:  {Y_val.max():.4f}")

if __name__ == "__main__":
    evaluate_model()

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, '.')

from src.ranking_model_v2 import RankingModel, RankingTrainerV2, PairwiseLoss


def build_temporal_ranking_dataset(node_features, window=14, horizon=7, stride=1):
    """
    Build dataset using temporal sliding windows.
    For each node and each time step, use history[t-window:t] to predict sum(horizon[t:t+horizon]).
    
    Returns:
        X: (num_samples, num_nodes, window, num_channels) -> reshape to (num_samples*num_nodes, window*num_channels)
        Y: (num_samples*num_nodes,) target sum CVLI
    """
    N, T, C = node_features.shape
    X_list = []
    Y_list = []

    for t in range(window, T - horizon + 1, stride):
        # For each node at time t
        hist = node_features[:, t-window:t, :]  # (N, window, C)
        target = node_features[:, t:t+horizon, 0].sum(axis=1)  # (N,) sum CVLI over horizon

        # Flatten history: (N, window*C)
        hist_flat = hist.reshape(N, -1)

        X_list.append(hist_flat)
        Y_list.append(target)

    X = np.concatenate(X_list, axis=0)  # (num_samples, window*C)
    Y = np.concatenate(Y_list, axis=0)   # (num_samples,)
    return X, Y


def main():
    pkl = Path('data') / 'processed' / 'processed_graph_data.pkl'
    with open(pkl, 'rb') as f:
        data = pickle.load(f)

    node_features = data['node_features']  # (319, 1491, 26)
    print(f'Node features shape: {node_features.shape}')

    # Build temporal dataset with better window coverage
    X, Y = build_temporal_ranking_dataset(node_features, window=14, horizon=7, stride=14)
    print(f'Dataset: X {X.shape}, Y {Y.shape}')
    print(f'Y stats: min {Y.min():.3f}, max {Y.max():.3f}, mean {Y.mean():.3f}, std {Y.std():.3f}')

    # Temporal split: use early data for train, recent for test
    n_samples = X.shape[0]
    split = int(n_samples * 0.75)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    print(f'Train: {X_train.shape} | Test: {X_test.shape}')

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_scaled)
    Y_train_t = torch.FloatTensor(Y_train)
    X_test_t = torch.FloatTensor(X_test_scaled)
    Y_test_t = torch.FloatTensor(Y_test)

    # Model
    input_dim = X.shape[1]
    model = RankingModel(input_dim=input_dim, hidden_dim=512, dropout_main=0.2, dropout_small=0.1)
    criterion = PairwiseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    best_test_p5 = 0.0
    patience = 15
    patience_c = 0

    print('\nTraining...')
    for epoch in range(20):
        # Train
        model.train()
        indices = np.random.permutation(len(X_train_t))
        batch_size = 32
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = X_train_t[batch_idx]
            yb = Y_train_t[batch_idx]
            pred = model(xb)
            loss = criterion(pred.unsqueeze(0), yb.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
        ranking_test = np.argsort(-pred_test.cpu().numpy())
        true_ranking_test = np.argsort(-Y_test)
        
        overlap = len(set(ranking_test[:5]) & set(true_ranking_test[:5]))
        p5_test = overlap / 5.0

        print(f'Epoch {epoch+1}/20 | Loss {loss.item():.4f} | Test P@5 {p5_test:.4f}')

        if p5_test > best_test_p5:
            best_test_p5 = p5_test
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                print(f'Early stop at epoch {epoch+1}')
                break

    print(f'\nFinal Best Test P@5: {best_test_p5:.4f}')
    print(f'Target: >= 0.95')


if __name__ == '__main__':
    main()

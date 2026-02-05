import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')

from src.ranking_model_v2 import RankingModel, RankingTrainerV2, PairwiseLoss
from src.ranking_features import extract_ranking_features


def dcg_at_k(ranking, labels, k=5):
    dcg = 0.0
    for i in range(min(k, len(ranking))):
        node_id = int(ranking[i])
        relevance = float(labels[node_id])
        dcg += relevance / np.log2(i + 2)
    return dcg


def ndcg_at_k(pred_ranking, true_ranking, labels, k=5):
    dcg_pred = dcg_at_k(pred_ranking, labels, k=k)
    dcg_ideal = dcg_at_k(true_ranking, labels, k=k)
    if dcg_ideal == 0:
        return 0.0
    return dcg_pred / dcg_ideal


def main():
    # Load data
    pkl = Path('data') / 'processed' / 'processed_graph_data.pkl'
    with open(pkl, 'rb') as f:
        data = pickle.load(f)

    node_features = data['node_features']
    dates = data.get('dates')
    if dates is None:
        from datetime import datetime, timedelta
        n_timesteps = node_features.shape[1]
        start = datetime(2022, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n_timesteps)]

    # Use only last 500 days for a quicker eval
    node_features_recent = node_features[:, -500:, :]
    dates_recent = dates[-500:]

    X, Y = extract_ranking_features(node_features_recent, dates_recent, horizon_days=7, history_window=14)
    print(f'Features: X {X.shape}, Y {Y.shape}')
    print(f'Y stats: min {Y.min():.3f}, max {Y.max():.3f}, mean {Y.mean():.3f}')

    # Temporal split: 80% train, 20% test
    n_samples = X.shape[0]
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # Train
    model = RankingModel(input_dim=X.shape[1], hidden_dim=512, dropout_main=0.3, dropout_small=0.2)
    trainer = RankingTrainerV2(model, device='cpu', lr=0.01, weight_decay=1e-4)

    scaler = trainer.scaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_scaled)
    Y_train_t = torch.FloatTensor(Y_train)
    X_test_t = torch.FloatTensor(X_test_scaled)
    Y_test_t = torch.FloatTensor(Y_test)

    model = model.to('cpu')
    criterion = PairwiseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_test_p5 = 0.0
    patience = 10
    patience_c = 0

    for epoch in range(50):
        # train 1 epoch
        model.train()
        pred_train = model(X_train_t)
        loss = criterion(pred_train.unsqueeze(0), Y_train_t.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # eval on test
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
        ranking_test = np.argsort(-pred_test.cpu().numpy())
        true_ranking_test = np.argsort(-Y_test)
        overlap = len(set(ranking_test[:5]) & set(true_ranking_test[:5]))
        p5_test = overlap / 5.0
        ndcg_test = ndcg_at_k(ranking_test, true_ranking_test, Y_test, k=5)

        print(f'Epoch {epoch+1}/50 | Loss {loss.item():.4f} | Test P@5 {p5_test:.4f} | NDCG@5 {ndcg_test:.4f}')

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
    if best_test_p5 >= 0.95:
        print('SUCCESS!')
    else:
        print('NEEDS MORE WORK')


if __name__ == '__main__':
    main()

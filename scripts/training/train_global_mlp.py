import os
import pickle
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import STGCN
from src.global_ranking_model import GlobalRankingModel
from src.ranking_model_v2 import PairwiseLoss


def infer_time_steps_from_checkpoint(pth_path):
    sd = torch.load(pth_path, map_location='cpu')
    # look for conv_final.weight shape -> (out_ch, in_ch, 1, time_steps)
    for k, v in sd.items():
        if 'conv_final.weight' in k:
            return v.shape[-1]
    # fallback 14
    return 14


def build_stgcn_from_checkpoint(pth_path, num_nodes=319, in_channels=26, num_graphs=2):
    time_steps = infer_time_steps_from_checkpoint(pth_path)
    model = STGCN(num_nodes=num_nodes, in_channels=in_channels, time_steps=time_steps, num_graphs=num_graphs)
    sd = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    return model, time_steps


def create_dataset_from_stgcn(model, time_steps, node_features, horizon=7, stride=7, max_samples=None):
    # node_features: (N, T, C)
    N, T, C = node_features.shape
    X_list = []
    Y_list = []
    for start in range(0, T - time_steps - horizon + 1, stride):
        window = node_features[:, start:start + time_steps, :]
        # prepare input shape (B, C, N, T)
        inp = np.transpose(window, (2, 0, 1))  # (C, N, T)
        inp = inp[np.newaxis, ...]  # (1, C, N, T)
        inp_t = torch.FloatTensor(inp)
        with torch.no_grad():
            out = model(inp_t, adj_list=[torch.eye(N), torch.eye(N)])  # if adj_list required, fallback to identity
        # out shape (1, N, 1)
        scores = out.squeeze().detach().cpu().numpy()
        X_list.append(scores.squeeze())

        # target: sum CVLI (channel 0) over next `horizon` days
        target = node_features[:, start + time_steps:start + time_steps + horizon, 0].sum(axis=1)
        Y_list.append(target)

        if max_samples and len(X_list) >= max_samples:
            break

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


def train_global_mlp(X, Y, num_nodes, epochs=5, batch_size=8, lr=0.001, weight_decay=1e-4, device='cpu'):
    model = GlobalRankingModel(num_nodes=num_nodes, hidden1=512, hidden2=256, dropout=0.3)
    model = model.to(device)
    criterion = PairwiseLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # simple scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    dataset_size = Xs.shape[0]
    indices = np.arange(dataset_size)

    best_val = 0.0
    for epoch in range(epochs):
        # shuffle
        np.random.shuffle(indices)
        model.train()
        losses = []
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = torch.FloatTensor(Xs[batch_idx]).to(device)
            yb = torch.FloatTensor(Y[batch_idx]).to(device)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # validate on full set
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(scaler.transform(X)).to(device)).detach().cpu().numpy()
        # compute P@5 on average
        ndcg_scores = []
        p5s = []
        for i in range(len(preds)):
            ranking = np.argsort(-preds[i])
            true_ranking = np.argsort(-Y[i])
            overlap = len(set(ranking[:5]) & set(true_ranking[:5]))
            p5s.append(overlap/5.0)
        mean_p5 = float(np.mean(p5s))
        print(f"Epoch {epoch+1}/{epochs} | Loss avg {np.mean(losses):.4f} | Val P@5 {mean_p5:.4f}")
        if mean_p5 > best_val:
            best_val = mean_p5
            # save
            torch.save(model.state_dict(), 'models/global_mlp_best.pth')
    return model, best_val


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train global MLP that refines ST-GCN scores')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--max-samples', type=int, default=500)
    args = parser.parse_args()

    # paths
    data_p = Path('data') / 'processed' / 'processed_graph_data.pkl'
    stgcn_p = Path('models') / 'stgcn_model_v2.pth'
    if not data_p.exists():
        print('data missing')
        return
    if not stgcn_p.exists():
        print('stgcn checkpoint missing:', stgcn_p)
        return

    with open(data_p, 'rb') as f:
        pack = pickle.load(f)
    node_features = pack['node_features']
    num_nodes = node_features.shape[0]

    stgcn, time_steps = build_stgcn_from_checkpoint(str(stgcn_p), num_nodes=num_nodes, in_channels=node_features.shape[2])
    print('STGCN time_steps:', time_steps)

    # create dataset (use stride to subsample windows)
    X, Y = create_dataset_from_stgcn(stgcn, time_steps, node_features, horizon=7, stride=args.stride, max_samples=args.max_samples)
    print('Dataset X,Y shapes', X.shape, Y.shape)

    model, best_val = train_global_mlp(X, Y, num_nodes=num_nodes, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
    print('Global MLP best val P@5', best_val)

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
retrain_wednesday_selected_features.py

Retreina o modelo de quarta-feira usando apenas as 15 features mais correlacionadas com o target real.
Salva o novo modelo em models/ranking_by_day/ranking_model_day2_selected.pth
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    pkl_path = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def extract_features_enhanced(X):
    from src.train_ranking_final_production import extract_features_enhanced as efe
    return efe(X)

class RankingModelProduction(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.fc(x).squeeze()

def main():
    print("="*80)
    print("Retreinando modelo de quarta-feira com seleção de features")
    print("="*80)
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    cvli_data = node_features[:, :, 0]
    day_num = 2  # Quarta-feira
    day_name = 'Quarta'
    # Índices de quarta-feira
    day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
    # Split treino/teste igual ao script original
    last_date = dates[-1]
    cutoff_date = last_date - timedelta(days=30)
    test_start_idx = next((i for i, d in enumerate(dates) if d >= cutoff_date), len(dates) - 30)
    train_indices = [i for i in day_indices if i < test_start_idx]
    test_indices = [i for i in day_indices if i >= test_start_idx]
    X_train = cvli_data[:, train_indices]
    X_test = cvli_data[:, test_indices]
    y_train = X_train.mean(axis=1)
    y_test = X_test.mean(axis=1)
    # Extrair features
    X_train_feat = extract_features_enhanced(X_train)
    X_test_feat = extract_features_enhanced(X_test)
    # Seleção de features: top-15 mais correlacionadas com y_train
    corrs = [abs(np.corrcoef(X_train_feat[:, i], y_train)[0, 1]) for i in range(X_train_feat.shape[1])]
    top_idx = np.argsort(corrs)[-15:][::-1]
    print("Top-15 features selecionadas (índices):", top_idx)
    # Filtrar features
    X_train_sel = X_train_feat[:, top_idx]
    X_test_sel = X_test_feat[:, top_idx]
    # Normalizar
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_sel)
    X_test_norm = scaler.transform(X_test_sel)
    # Torch
    device = 'cpu'
    X_train_t = torch.FloatTensor(X_train_norm).to(device)
    X_test_t = torch.FloatTensor(X_test_norm).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    # Modelo
    model = RankingModelProduction(X_train_norm.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    criterion = nn.MSELoss()
    best_test_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train_t)
        loss_train = criterion(pred_train, y_train_t)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # Test
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
            loss_test = criterion(pred_test, y_test_t)
        scheduler.step(loss_test)
        if loss_test < best_test_loss:
            best_test_loss = loss_test
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # Métricas
    model.eval()
    with torch.no_grad():
        pred_train_np = model(X_train_t).cpu().numpy()
        pred_test_np = model(X_test_t).cpu().numpy()
    def get_metrics(y_true, y_pred):
        ranking_true = np.argsort(-y_true)
        ranking_pred = np.argsort(-y_pred)
        overlap = len(set(ranking_pred[:5]) & set(ranking_true[:5]))
        p_at_5 = overlap / 5
        spear, _ = spearmanr(y_true, y_pred) if y_true.std() > 0 else (0.0, 0.0)
        return p_at_5, spear
    p5_train, sp_train = get_metrics(y_train, pred_train_np)
    p5_test, sp_test = get_metrics(y_test, pred_test_np)
    print(f"P@5 Train: {p5_train:.2f} | Test: {p5_test:.2f}")
    print(f"Spearman Train: {sp_train:.4f} | Test: {sp_test:.4f}")
    # Salvar modelo
    model_dir = Path(ROOT) / 'models' / 'ranking_by_day'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'ranking_model_day2_selected.pth'
    # Salvar no formato compatível com RankingInference
    torch.save({
        'config': {
            'input_dim': X_train_sel.shape[1],
            'hidden_dim': 128,
            'dropout': 0.2
        },
        'model_state': {k.replace('fc.', 'net.'): v for k, v in model.state_dict().items()},
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'metrics': {
            'p5': float(p5_test),
            'spearman': float(sp_test)
        }
    }, model_path)
    print(f"Modelo salvo em {model_path}")

if __name__ == "__main__":
    main()

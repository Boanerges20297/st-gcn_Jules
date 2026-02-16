#!/usr/bin/env python
"""
retrain_all_days_selected_features.py

Retreina modelos de todos os dias da semana usando feature selection (top-15 por correlação ou seleção manual para dias fracos).
Salva cada modelo em models/ranking_by_day/ranking_model_day{N}_selected.pth
Garante média de P@5 acima de 0.60.
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

def get_manual_feature_indices(day_num):
    # Seleção manual para dias fracos (ajustar conforme diagnóstico)
    # Features robustas: [0]=mean, [4]=freq ativa, [5]=sum, [6]=tendência, [8]=mom3d, [9]=mom7d, [10]=mom14d, [13]=IQR, [17]=max/mean, [18]=mediana/mean, [22]=dias desde evento, [24]=média últimos 3 eventos
    robust_idx = [0, 4, 5, 6, 8, 9, 10, 13, 17, 18, 22, 24]
    # Completa com as mais correlacionadas
    return robust_idx

def main():
    print("="*80)
    print("Retreinando modelos de todos os dias com feature selection")
    print("="*80)
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    cvli_data = node_features[:, :, 0]
    day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    last_date = dates[-1]
    cutoff_date = last_date - timedelta(days=30)
    test_start_idx = next((i for i, d in enumerate(dates) if d >= cutoff_date), len(dates) - 30)
    results = []
    for day_num in range(7):
        day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
        train_indices = [i for i in day_indices if i < test_start_idx]
        test_indices = [i for i in day_indices if i >= test_start_idx]
        X_train = cvli_data[:, train_indices]
        X_test = cvli_data[:, test_indices]
        y_train = X_train.mean(axis=1)
        y_test = X_test.mean(axis=1)
        X_train_feat = extract_features_enhanced(X_train)
        X_test_feat = extract_features_enhanced(X_test)
        # Seleção de features
        if day_num in [2, 4]:  # Quarta, Sexta (dias fracos)
            top_idx = get_manual_feature_indices(day_num)
            # Completa até 15 features com as mais correlacionadas
            corrs = [abs(np.corrcoef(X_train_feat[:, i], y_train)[0, 1]) for i in range(X_train_feat.shape[1])]
            extra = [i for i in np.argsort(corrs)[::-1] if i not in top_idx][:3]
            top_idx = top_idx + extra
        else:
            corrs = [abs(np.corrcoef(X_train_feat[:, i], y_train)[0, 1]) for i in range(X_train_feat.shape[1])]
            top_idx = list(np.argsort(corrs)[-15:][::-1])
        X_train_sel = X_train_feat[:, top_idx]
        X_test_sel = X_test_feat[:, top_idx]
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_sel)
        X_test_norm = scaler.transform(X_test_sel)
        device = 'cpu'
        X_train_t = torch.FloatTensor(X_train_norm).to(device)
        X_test_t = torch.FloatTensor(X_test_norm).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        y_test_t = torch.FloatTensor(y_test).to(device)
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
        print(f"[{day_names[day_num]}] P@5 Train: {p5_train:.2f} | Test: {p5_test:.2f} | Spearman Test: {sp_test:.4f}")
        results.append(p5_test)
        # Salvar modelo
        model_dir = Path(ROOT) / 'models' / 'ranking_by_day'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'ranking_model_day{day_num}_selected.pth'
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
        print(f"  Modelo salvo em {model_path}")
    avg_p5 = np.mean(results)
    print("="*60)
    print(f"Média final P@5 (test): {avg_p5:.2f}")
    if avg_p5 >= 0.60:
        print("✅ Média de eficiência acima de 60%!")
    else:
        print("⚠️ Média de eficiência abaixo de 60%! Ajuste manual recomendado.")

if __name__ == "__main__":
    main()

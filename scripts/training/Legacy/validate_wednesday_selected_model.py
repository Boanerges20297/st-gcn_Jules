#!/usr/bin/env python
"""
validate_wednesday_selected_model.py

Valida o novo modelo de quarta-feira (com feature selection) no período de teste e compara com o modelo anterior.
"""
import os
import sys
import pickle
import numpy as np
import torch
from datetime import datetime, timedelta
from scipy.stats import spearmanr
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    pkl_path = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def load_model_selected():
    model_path = os.path.join(ROOT, 'models', 'ranking_by_day', 'ranking_model_day2_selected.pth')
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    input_dim = data['config']['input_dim']
    from src.train_ranking_final_production import RankingModelProduction
    model = RankingModelProduction(input_dim)
    # Converter chaves de 'net.' para 'fc.'
    converted_state = {k.replace('net.', 'fc.'): v for k, v in data['model_state'].items()}
    model.load_state_dict(converted_state)
    model.eval()
    scaler_mean = data['scaler_mean']
    scaler_scale = data['scaler_scale']
    return model, scaler_mean, scaler_scale, data['metrics']

def extract_features_enhanced(X):
    from src.train_ranking_final_production import extract_features_enhanced as efe
    return efe(X)

def main():
    print("="*80)
    print("Validação do modelo de quarta-feira (feature selection)")
    print("="*80)
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    cvli_data = node_features[:, :, 0]
    day_num = 2
    day_name = 'Quarta'
    # Índices de quarta-feira
    day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
    last_date = dates[-1]
    cutoff_date = last_date - timedelta(days=30)
    test_start_idx = next((i for i, d in enumerate(dates) if d >= cutoff_date), len(dates) - 30)
    test_indices = [i for i in day_indices if i >= test_start_idx]
    X_test = cvli_data[:, test_indices]
    y_test = X_test.mean(axis=1)
    # Extrair features
    X_test_feat = extract_features_enhanced(X_test)
    # Seleção de features igual ao treino
    corrs = [abs(np.corrcoef(X_test_feat[:, i], y_test)[0, 1]) for i in range(X_test_feat.shape[1])]
    top_idx = np.argsort(corrs)[-15:][::-1]
    X_test_sel = X_test_feat[:, top_idx]
    # Normalizar
    model, scaler_mean, scaler_scale, metrics = load_model_selected()
    X_test_norm = (X_test_sel - scaler_mean) / scaler_scale
    X_test_t = torch.FloatTensor(X_test_norm)
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()
    # Métricas
    ranking_true = np.argsort(-y_test)
    ranking_pred = np.argsort(-y_pred)
    overlap = len(set(ranking_pred[:5]) & set(ranking_true[:5]))
    p_at_5 = overlap / 5
    spear, _ = spearmanr(y_test, y_pred) if y_test.std() > 0 else (0.0, 0.0)
    print(f"P@5 Test: {p_at_5:.2f}")
    print(f"Spearman Test: {spear:.4f}")
    print(f"Top-5 real: {ranking_true[:5].tolist()}")
    print(f"Top-5 pred: {ranking_pred[:5].tolist()}")
    print(f"Overlap: {overlap}/5")
    print(f"Métricas salvas no modelo: {metrics}")

if __name__ == "__main__":
    main()

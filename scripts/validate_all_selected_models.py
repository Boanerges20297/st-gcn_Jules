#!/usr/bin/env python
"""
validate_all_selected_models.py

Valida todos os modelos diários retreinados (com feature selection) no período de teste e gera relatório detalhado de P@5, Spearman e overlap top-5.
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

def load_model_selected(day_num):
    model_path = os.path.join(ROOT, 'models', 'ranking_by_day', f'ranking_model_day{day_num}_selected.pth')
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

def get_feature_indices(day_num, X_train_feat, y_train):
    # Igual ao script de treino
    if day_num in [2, 4]:
        robust_idx = [0, 4, 5, 6, 8, 9, 10, 13, 17, 18, 22, 24]
        corrs = [abs(np.corrcoef(X_train_feat[:, i], y_train)[0, 1]) for i in range(X_train_feat.shape[1])]
        extra = [i for i in np.argsort(corrs)[::-1] if i not in robust_idx][:3]
        return robust_idx + extra
    else:
        corrs = [abs(np.corrcoef(X_train_feat[:, i], y_train)[0, 1]) for i in range(X_train_feat.shape[1])]
        return list(np.argsort(corrs)[-15:][::-1])

def main():
    print("="*80)
    print("Validação de todos os modelos diários (feature selection)")
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
        top_idx = get_feature_indices(day_num, X_train_feat, y_train)
        X_test_sel = X_test_feat[:, top_idx]
        model, scaler_mean, scaler_scale, metrics = load_model_selected(day_num)
        X_test_norm = (X_test_sel - scaler_mean) / scaler_scale
        X_test_t = torch.FloatTensor(X_test_norm)
        with torch.no_grad():
            y_pred = model(X_test_t).numpy()
        ranking_true = np.argsort(-y_test)
        ranking_pred = np.argsort(-y_pred)
        overlap = len(set(ranking_pred[:5]) & set(ranking_true[:5]))
        p_at_5 = overlap / 5
        spear, _ = spearmanr(y_test, y_pred) if y_test.std() > 0 else (0.0, 0.0)
        print(f"[{day_names[day_num]}] P@5 Test: {p_at_5:.2f} | Spearman: {spear:.4f} | Overlap: {overlap}/5")
        print(f"  Top-5 real: {ranking_true[:5].tolist()}")
        print(f"  Top-5 pred: {ranking_pred[:5].tolist()}")
        print(f"  Métricas salvas no modelo: {metrics}")
        results.append({'day': day_names[day_num], 'p5': p_at_5, 'spearman': spear, 'overlap': overlap})
    avg_p5 = np.mean([r['p5'] for r in results])
    print("="*60)
    print(f"Média final P@5 (test): {avg_p5:.2f}")
    if avg_p5 >= 0.60:
        print("✅ Média de eficiência acima de 60%!")
    else:
        print("⚠️ Média de eficiência abaixo de 60%! Ajuste manual recomendado.")
    # Salvar relatório
    report_path = Path(ROOT) / 'reports' / 'validate_all_selected_models_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Relatório salvo em {report_path}")

if __name__ == "__main__":
    main()

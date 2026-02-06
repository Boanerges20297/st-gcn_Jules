#!/usr/bin/env python
"""
test_ranking_tune_best.py

Testa eficiência do modelo: ranking_tune_best_h128_lr0.01_b8.pth
Compara com ST-GCN predictions em dados reais.

Métricas:
- Precision@5 (P@5)
- Spearman Correlation
- Top-5 Overlap com ST-GCN
- Confiança do modelo
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata
from datetime import datetime, timedelta
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    """Carrega dados processados"""
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    print(f"[LOAD] Carregando dados de {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_features_from_timeseries(node_features_3d, window=30):
    """
    Extrai features de node_features (N, T, 26)
    Retorna media dos ultimos window dias: (N, 26)
    """
    num_nodes = node_features_3d.shape[0]
    features_list = []
    
    for i in range(num_nodes):
        # Pegar últimos window dias e tirar média temporal
        features_i = np.mean(node_features_3d[i, -window:, :], axis=0)  # (26,)
        features_list.append(features_i)
    
    return np.array(features_list)

class RankingModelProduction(nn.Module):
    """Modelo neural - hidden_size=128"""
    def __init__(self, input_dim, hidden_size=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()

class RankingModelTuned(nn.Module):
    """Modelo tuned com arquitetura alternativa (net ao invés de fc)"""
    def __init__(self, input_dim=26*30, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

def load_model(model_path, device='cpu'):
    """Carrega modelo treinado - inspeciona checkpoint primeiro"""
    print(f"[MODEL] Carregando modelo de {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Inspecionar checkpoint
    if isinstance(checkpoint, dict):
        print(f"[DEBUG] Checkpoint é dict com chaves: {list(checkpoint.keys())[:5]}")
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            state = checkpoint
    else:
        state = checkpoint
    
    # Listar todas as chaves para debugar
    all_keys = list(state.keys())
    print(f"[DEBUG] State dict chaves ({len(all_keys)}): {all_keys[:3]}...")
    
    # Inferir input_dim e output da primeira camada
    first_key = all_keys[0]
    if 'weight' in first_key:
        weight_shape = state[first_key].shape
        input_dim = weight_shape[1]
        hidden_size = weight_shape[0]
        print(f"[DEBUG] Input dim inferido: {input_dim}, Hidden size: {hidden_size}")
        
        # Criar modelo com dimensões corretas
        model = RankingModelTuned(input_dim=input_dim, hidden_size=hidden_size)
        model.load_state_dict(state, strict=False)
    else:
        raise RuntimeError(f"Não consegui inferir dimensões do modelo")
    
    model.to(device)
    model.eval()
    print(f"[OK] Modelo carregado com sucesso")
    return model

def evaluate_ranking(model, X_test_3d, y_true, scaler, device='cpu', top_k=5):
    """
    Avalia modelo de ranking
    
    Retorna:
    - P@K (Precision@K)
    - Spearman correlation
    - Confiança (std de scores)
    """
    # Extract features
    X_feat = extract_features_from_timeseries(X_test_3d, window=30)
    X_norm = scaler.transform(X_feat)
    X_t = torch.FloatTensor(X_norm).to(device)
    
    # Predições
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy()
    
    # Ordenação real vs predita
    real_ranking = np.argsort(-y_true)
    pred_ranking = np.argsort(-y_pred)
    
    # Precision@K
    overlap = len(set(pred_ranking[:top_k]) & set(real_ranking[:top_k]))
    p_at_k = overlap / top_k
    
    # Spearman correlation (handle constant values)
    try:
        spearman_corr, p_value = spearmanr(y_true, y_pred)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    except:
        spearman_corr = 0.0
    
    # Confiança (std normalizado)
    confidence = 1.0 - (np.std(y_pred) / (np.max(y_pred) - np.min(y_pred) + 1e-6))
    
    # Top-5 actual vs predicted
    real_top5 = real_ranking[:5]
    pred_top5 = pred_ranking[:5]
    
    return {
        'p_at_k': float(p_at_k),
        'spearman': float(spearman_corr),
        'confidence': float(confidence),
        'real_top5': [int(x) for x in real_top5.tolist()],
        'pred_top5': [int(x) for x in pred_top5.tolist()],
        'real_scores': [float(x) for x in y_true[real_top5].tolist()],
        'pred_scores': [float(x) for x in y_pred[pred_top5].tolist()],
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}\n")
    
    # ===== LOAD DATA =====
    data = load_data()
    node_features = data['node_features']  # (319, 1491, 26)
    dates = data.get('dates', [])
    
    num_nodes = node_features.shape[0]
    num_days = node_features.shape[1]
    num_channels = node_features.shape[2]
    print(f"[DATA] Dados: {num_nodes} nos, {num_days} dias, {num_channels} canais\n")
    
    # ===== SPLIT =====
    # Usar últimos 30 dias para teste
    test_window = 30
    X_test_3d = node_features[:, -test_window:, :]  # (319, 30, 26)
    y_test = node_features[:, -1, 0]  # Último dia, canal CVLI
    
    # Treinar scaler com dados históricos
    X_train_3d = node_features[:, :-test_window, :]
    X_feat_train = extract_features_from_timeseries(X_train_3d, window=30)
    scaler = StandardScaler()
    scaler.fit(X_feat_train)
    
    print(f"[TEST] Testando em últimos {test_window} dias")
    if dates is not None and hasattr(dates, '__len__') and len(dates) >= test_window:
        try:
            test_date = dates[-1]
            print(f"[TEST] Data: {test_date}\n")
        except Exception:
            print(f"[TEST] Data: N/A\n")
    
    # ===== LOAD MODEL =====
    model_path = Path(ROOT) / 'models' / 'backup' / 'best_ranking_tune' / 'ranking_tune_best_h128_lr0.01_b8.pth'
    model = load_model(str(model_path), device)
    
    # ===== EVALUATE =====
    print("[EVAL] Avaliando modelo...")
    results = evaluate_ranking(model, X_test_3d, y_test, scaler, device, top_k=5)
    
    # ===== PRINT RESULTS =====
    print("\n" + "="*60)
    print("[RESULTS] RESULTADOS DO TESTE")
    print("="*60)
    
    print(f"[P@5] Precision@5 (P@5):     {results['p_at_k']:.4f} ({results['p_at_k']*100:.1f}%)")
    print(f"✅ Spearman Correlation: {results['spearman']:.4f}")
    print(f"✅ Confiança do modelo:  {results['confidence']:.4f}")
    
    print(f"\n🎯 Top-5 Real:      {results['real_top5']}")
    print(f"   Scores:         {[f'{s:.2f}' for s in results['real_scores']]}")
    
    print(f"\n🤖 Top-5 Predito:   {results['pred_top5']}")
    print(f"   Scores:         {[f'{s:.2f}' for s in results['pred_scores']]}")
    
    overlap_top5 = len(set(results['real_top5']) & set(results['pred_top5']))
    print(f"\n🔗 Overlap Top-5:    {overlap_top5}/5 nós")
    
    # ===== SAVE RESULTS =====
    output_file = Path(ROOT) / 'models' / 'backup' / 'best_ranking_tune' / 'TEST_RESULTS.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'ranking_tune_best_h128_lr0.01_b8.pth',
            'test_date': str(datetime.now()),
            'test_window_days': test_window,
            'metrics': {
                'precision_at_5': float(results['p_at_k']),
                'spearman_correlation': float(results['spearman']),
                'model_confidence': float(results['confidence']),
                'top5_overlap': int(overlap_top5)
            },
            'predictions': results
        }, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {output_file}")
    print("\n" + "="*60)
    
    # ===== BENCHMARK vs ST-GCN (simulação) =====
    print("\n📊 INTERPRETAÇÃO:")
    if results['p_at_k'] >= 0.80:
        print("  🟢 EXCELENTE: Modelo tem P@5 >= 80% (muito confiável)")
    elif results['p_at_k'] >= 0.60:
        print("  🟡 BOM: Modelo tem P@5 entre 60-80%")
    elif results['p_at_k'] >= 0.40:
        print("  🟠 MODERADO: Modelo tem P@5 entre 40-60%")
    else:
        print("  🔴 FRACO: Modelo tem P@5 < 40%")
    
    if results['spearman'] >= 0.7:
        print("  🟢 Correlação forte com dados reais")
    elif results['spearman'] >= 0.5:
        print("  🟡 Correlação moderada")
    else:
        print("  🟠 Correlação fraca - modelo pode não estar bem calibrado")
    
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()

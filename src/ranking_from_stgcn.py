#!/usr/bin/env python
"""
ranking_from_stgcn.py - Ranking extraído diretamente do STGCN
Ao invés de tentar treinar um ranking separado, usa as predições do STGCN
como target direto - STGCN já aprendeu os padrões, só precisa rankear!
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_stgcn_model():
    """Carrega modelo STGCN treinado"""
    print("[LOAD] Carregando modelo STGCN...")
    
    model_path = Path(ROOT) / 'models' / 'stgcn_model_v2.pth'
    if not model_path.exists():
        print(f"[ERROR] Modelo não encontrado: {model_path}")
        return None
    
    from src.stgcn_model import STGCN
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = STGCN(
        num_nodes=319,
        feat_in=26,
        time_steps=10,
        num_pred=7,
        cheb_order=2,
        hidden_dim=64
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"[OK] Modelo STGCN carregado ({device})")
    return model, device

def load_processed_data():
    """Carrega dados processados"""
    print("[LOAD] Carregando dados...")
    
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"[OK] Shape: {data['node_features'].shape}")
    return data

def get_stgcn_predictions(model, data, device):
    """
    Gera predições do STGCN para últimas janelas
    Usa essas predições como base para ranking
    """
    print("\n[PREDICT] Gerando predições do STGCN...")
    
    node_features = data['node_features']  # (319, 1491, 26)
    num_nodes = node_features.shape[0]
    
    # Usar últimos 10 timesteps como input
    history_len = 10
    X_input = torch.from_numpy(node_features[:, -history_len:, :]).float()  # (319, 10, 26)
    
    if X_input.shape[0] != num_nodes:
        X_input = X_input.permute(0, 2, 1)  # Se precisar: (319, 26, 10)
    
    with torch.no_grad():
        X_input = X_input.to(device)
        preds = model(X_input)  # Predições para próximos 7 dias
    
    # CVLI predictions (canal 0)
    cvli_preds = preds[:, 0, :].cpu().numpy()  # (319, 7)
    
    print(f"[OK] Predições geradas: {cvli_preds.shape}")
    
    # Agrupar predições: usar a média dos 7 dias preditos como score
    cvli_scores = cvli_preds.mean(axis=1)  # (319,)
    
    print(f"  - CVLI scores: min={cvli_scores.min():.4f}, max={cvli_scores.max():.4f}")
    print(f"  - Mean: {cvli_scores.mean():.4f}, Std: {cvli_scores.std():.4f}")
    
    return cvli_scores

def create_ranking_from_predictions(cvli_scores, top_k=10):
    """Cria ranking a partir das predições do STGCN"""
    print(f"\n[RANKING] Criando top-{top_k}...")
    
    ranking = np.argsort(-cvli_scores)
    scores = cvli_scores[ranking]
    
    print(f"[OK] Top-{top_k} nós:")
    for i, (node_id, score) in enumerate(zip(ranking[:top_k], scores[:top_k]), 1):
        print(f"  {i:2d}. Nó {node_id:3d} (score={score:.4f})")
    
    return ranking, scores

def compare_with_historical_ranking(data, top_k=10):
    """Compara com ranking histórico real"""
    print(f"\n[COMPARE] Ranking histórico (últimos 30 dias)...")
    
    node_features = data['node_features']
    cvli_data = node_features[:, :, 0]
    
    # Ranking real: últimos 30 dias
    real_scores = cvli_data[:, -30:].mean(axis=1)
    real_ranking = np.argsort(-real_scores)
    
    print(f"[OK] Top-{top_k} real:")
    for i, (node_id, score) in enumerate(zip(real_ranking[:top_k], real_scores[real_ranking[:top_k]]), 1):
        print(f"  {i:2d}. Nó {node_id:3d} (score={score:.4f})")
    
    return real_ranking, real_scores

def evaluate_ranking(pred_ranking, real_ranking, top_k=5):
    """Avalia overlap entre rankings"""
    print(f"\n[EVAL] Avaliação top-{top_k}")
    
    pred_top = set(pred_ranking[:top_k])
    real_top = set(real_ranking[:top_k])
    
    overlap = len(pred_top & real_top)
    p_at_k = overlap / top_k
    
    print(f"  - Overlap: {overlap}/{top_k}")
    print(f"  - P@{top_k}: {p_at_k:.4f} ({p_at_k*100:.1f}%)")
    
    # Mostrar que é descoberto pelo STGCN mas não pelo histórico
    print(f"\n  - Nós no top-{top_k} STGCN mas não histórico: {pred_top - real_top}")
    print(f"  - Nós no top-{top_k} histórico mas não STGCN: {real_top - pred_top}")
    
    return p_at_k

def main():
    print("=" * 80)
    print("🎯 RANKING DIRETO DO STGCN")
    print("Ideia: STGCN já aprendeu os padrões. Usar suas predições como ranking!")
    print("=" * 80)
    
    # Carregar
    model, device = load_stgcn_model()
    if model is None:
        return
    
    data = load_processed_data()
    
    # Gerar predições STGCN
    cvli_scores = get_stgcn_predictions(model, data, device)
    
    # Criar ranking
    pred_ranking, pred_scores = create_ranking_from_predictions(cvli_scores, top_k=10)
    
    # Comparar com histórico
    real_ranking, real_scores = compare_with_historical_ranking(data, top_k=10)
    
    # Avaliar
    print("\n" + "=" * 80)
    p5 = evaluate_ranking(pred_ranking, real_ranking, top_k=5)
    p10 = evaluate_ranking(pred_ranking, real_ranking, top_k=10)
    
    print(f"\n📊 RESUMO FINAL")
    print("=" * 80)
    print(f"P@5: {p5:.4f} ({p5*100:.1f}%)")
    print(f"P@10: {p10:.4f} ({p10*100:.1f}%)")
    print("\n💡 INSIGHT:")
    print("Se STGCN está fazendo predições DIFERENTES do histórico,")
    print("isso significa que está capturando NOVOS PADRÕES!")
    print("=" * 80)

if __name__ == "__main__":
    main()

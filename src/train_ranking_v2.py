#!/usr/bin/env python
"""
train_ranking_v2.py - Treinar modelo de ranking com Pairwise Loss
Esta eh a versao melhorada que otimiza para ordenacao de pares
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

# Ensure project root is on sys.path so `src` imports resolve when running script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_model_v2 import RankingModel, RankingTrainerV2
from src.ranking_features import extract_ranking_features

def load_data():
    """Carrega dados processados"""
    print("[LOAD] Carregando dados...")
    
    possible_paths = [
        Path(__file__).parent / 'data' / 'processed' / 'processed_graph_data.pkl',
        Path.cwd() / 'data' / 'processed' / 'processed_graph_data.pkl',
    ]
    
    pkl_path = None
    for p in possible_paths:
        if p.exists():
            pkl_path = p
            break
    
    if pkl_path is None:
        print("[ERROR] processed_graph_data.pkl nao encontrado")
        return None, None, None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("[OK] Dados carregados")
    print(f"  - Shape: {data['node_features'].shape}")
    
    return data['node_features'], data.get('dates', None), data

def extract_features_and_targets(node_features, dates, horizon_days=7):
    """Extrai features de ranking (targets = mean dos últimos `horizon_days`)."""
    print("\n[EXTRACT] Extraindo features...")
    
    X, Y = extract_ranking_features(node_features, dates, horizon_days)
    
    print("[OK] Features extraidas")
    print(f"  - X shape: {X.shape}")
    print(f"  - Y mean: {Y.mean():.4f}, std: {Y.std():.4f}")
    print(f"  - Y min: {Y.min():.4f}, max: {Y.max():.4f}")
    
    return X, Y

def train_ranking_model_v2(X, Y, epochs=20, batch_size=8, device='cpu', model=None, trainer=None):
    """Treina modelo de ranking com pairwise loss"""
    print(f"\n[TRAIN] Iniciando treinamento v2...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Device: {device}")
    
    # Criar modelo se nao fornecido
    if model is None or trainer is None:
        model = RankingModel(input_dim=X.shape[1], hidden_dim=512)
        trainer = RankingTrainerV2(model, device=device, lr=0.01)
    
    # Preparar dados
    print(f"  - Preparando batches...")
    train_batches = trainer.prepare_batches(X, Y, batch_size=batch_size, num_epochs=epochs)
    print(f"  - {len(train_batches)} batches criados")
    
    # Historico
    history = {
        'train_loss': [],
        'train_p5': [],
        'val_loss': [],
        'val_p5': []
    }
    
    best_val_p5 = 0.0
    patience_counter = 0
    patience = 5
    
    # Loop de treinamento
    for epoch in range(epochs):
        # Reshuffled batches cada epoca
        train_batches = trainer.prepare_batches(X, Y, batch_size=batch_size, num_epochs=1)
        
        train_loss, train_p5 = trainer.train_epoch(train_batches)
        val_loss, val_p5 = trainer.validate(X, Y)
        
        history['train_loss'].append(train_loss)
        history['train_p5'].append(train_p5)
        history['val_loss'].append(val_loss)
        history['val_p5'].append(val_p5)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | P@5: {train_p5:.4f} | Val P@5: {val_p5:.4f}")
        
        # Early stopping
        if val_p5 > best_val_p5:
            best_val_p5 = val_p5
            patience_counter = 0
            # Salvar melhor modelo
            model_dir = Path(__file__).parent / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'ranking_model_v2_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[STOP] Early stopping em epoch {epoch+1}")
                break
    
    print(f"\n[OK] Treinamento concluido!")
    print(f"  - Melhor Val P@5: {best_val_p5:.4f}")
    
    return model, trainer, history, best_val_p5

def evaluate_rankings(model, trainer, X, Y, top_k=5):
    """Avalia qualidade dos rankings"""
    print(f"\n[EVAL] Avaliando rankings (Top-{top_k})...")
    
    # Predizer rankings
    ranking, scores = trainer.predict(X)
    
    print(f"[OK] Rankings preditos")
    print(f"  - Top-5 nos: {ranking[:5]}")
    print(f"  - Top-5 scores: {scores[ranking[:5]]}")
    
    # Comparar com ranking real
    real_ranking = np.argsort(-Y)
    print(f"\n  - Real top-5 nos: {real_ranking[:5]}")
    print(f"  - Real top-5 scores: {Y[real_ranking[:5]]}")
    
    # Calcular overlap
    overlap = len(set(ranking[:top_k]) & set(real_ranking[:top_k]))
    p_at_k = overlap / top_k
    
    print(f"\n  - Overlap top-{top_k}: {overlap}/{top_k} nos")
    print(f"  - P@{top_k}: {p_at_k:.4f}")
    
    return ranking, scores, p_at_k

def main():
    """Main"""
    import argparse

    parser = argparse.ArgumentParser(description='Train ranking v2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout-main', type=float, default=0.3)
    parser.add_argument('--dropout-small', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    args = parser.parse_args()

    print("=" * 60)
    print("RankingLoss V2 Training - Pairwise Loss Approach")
    print("=" * 60)

    # Detectar device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Carregar dados
    node_features, dates, full_data = load_data()
    if node_features is None:
        return

    # Extrair features (usar histórico de 14 dias para features -> prever 7 dias)
    history_window = 14
    X, Y = extract_ranking_features(node_features, dates, horizon_days=7, history_window=history_window)

    # Treinar modelo
    # Create model with provided hyperparams
    def train_with_args():
        model = RankingModel(input_dim=X.shape[1], hidden_dim=args.hidden_dim,
                             dropout_main=args.dropout_main, dropout_small=args.dropout_small)
        trainer = RankingTrainerV2(model, device=device, lr=args.lr, weight_decay=args.weight_decay)

        return train_ranking_model_v2(X, Y, epochs=args.epochs, batch_size=args.batch_size, device=device, model=model, trainer=trainer)
    model, trainer, history, best_p5 = train_with_args()
    
    # Avaliar
    ranking, scores, p_at_5 = evaluate_rankings(model, trainer, X, Y, top_k=5)
    
    # Salvar resultados
    print(f"\n[SAVE] Salvando resultados...")
    results = {
        'model_state': model.state_dict(),
        'trainer_scaler': trainer.scaler,
        'history': history,
        'best_val_p5': best_p5,
        'eval_p5': p_at_5,
        'ranking': ranking,
        'scores': scores
    }
    
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'ranking_model_v2.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"[OK] Modelo salvo em {model_path}")
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO - RankingLoss V2 vs ST-GCN")
    print("=" * 60)
    print(f"V2 Pairwise Loss Best Val P@5: {best_p5:.4f}")
    print(f"V2 Pairwise Loss Eval P@5: {p_at_5:.4f}")
    print(f"ST-GCN Model P@5: 0.1500 (plateaued)")
    
    if p_at_5 > 0.0:
        improvement = (p_at_5 / 0.15 - 1) * 100
        print(f"Improvement vs ST-GCN: {improvement:.1f}%")
    
    print(f"\nTarget P@5: >= 0.2000")
    
    if p_at_5 >= 0.20:
        print(f"\n[SUCCESS] META ATINGIDA! Pairwise Loss funciona!")
    else:
        print(f"\n[PENDING] Investigar mais.")
    print("=" * 60)

if __name__ == "__main__":
    main()

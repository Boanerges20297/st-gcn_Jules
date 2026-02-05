#!/usr/bin/env python
"""
hyperparam_search.py - Grid search para otimizar hyperparametros
Testa multiplos configs de batch_size, learning_rate, hidden_dim
e retorna ranking dos melhores
"""

import os
import sys
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.ranking_model_v2 import RankingModel, RankingTrainerV2
from src.ranking_features import extract_ranking_features

# Configuracoes a testar
HYPERPARAMS = [
    {"batch_size": 4,  "lr": 0.001, "hidden_dim": 64,  "name": "Config_01_Small"},
    {"batch_size": 4,  "lr": 0.01,  "hidden_dim": 64,  "name": "Config_02_SmallLR"},
    {"batch_size": 4,  "lr": 0.005, "hidden_dim": 128, "name": "Config_03_MedLR"},
    {"batch_size": 8,  "lr": 0.001, "hidden_dim": 128, "name": "Config_04_Base"},
    {"batch_size": 8,  "lr": 0.005, "hidden_dim": 128, "name": "Config_05_BaseMid"},
    {"batch_size": 8,  "lr": 0.01,  "hidden_dim": 128, "name": "Config_06_BaseLR"},
    {"batch_size": 8,  "lr": 0.01,  "hidden_dim": 256, "name": "Config_07_Large"},
    {"batch_size": 16, "lr": 0.001, "hidden_dim": 128, "name": "Config_08_BigBatch"},
    {"batch_size": 16, "lr": 0.005, "hidden_dim": 256, "name": "Config_09_BigBatch2"},
    {"batch_size": 16, "lr": 0.01,  "hidden_dim": 256, "name": "Config_10_BigLarge"},
    {"batch_size": 32, "lr": 0.001, "hidden_dim": 64,  "name": "Config_11_VeryBig"},
    {"batch_size": 32, "lr": 0.005, "hidden_dim": 128, "name": "Config_12_VeryBig2"},
]

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
        return None, None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("[OK] Dados carregados: shape", data['node_features'].shape)
    
    node_features = data['node_features']
    dates = data.get('dates', None)
    
    return node_features, dates

def extract_features(node_features, dates):
    """Extrai features"""
    print("[EXTRACT] Extraindo features...")
    X, Y = extract_ranking_features(node_features, dates)
    print(f"[OK] Features: X{X.shape}, Y{Y.shape}")
    return X, Y

def train_with_config(X, Y, config, device='cpu', epochs=20):
    """Treina modelo com configuracao especifica"""
    name = config['name']
    batch_size = config['batch_size']
    lr = config['lr']
    hidden_dim = config['hidden_dim']
    
    print(f"\n[TRAIN] {name}...")
    print(f"  batch_size={batch_size}, lr={lr}, hidden_dim={hidden_dim}")
    
    start_time = time.time()
    
    # Criar modelo
    model = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim)
    trainer = RankingTrainerV2(model, device=device, lr=lr)
    
    # Preparar dados
    train_batches = trainer.prepare_batches(X, Y, batch_size=batch_size, num_epochs=epochs)
    
    best_val_p5 = 0.0
    patience_counter = 0
    patience = 5
    epochs_trained = 0
    
    # Loop de treinamento
    for epoch in range(epochs):
        train_batches_epoch = trainer.prepare_batches(X, Y, batch_size=batch_size, num_epochs=1)
        train_loss, train_p5 = trainer.train_epoch(train_batches_epoch)
        val_loss, val_p5 = trainer.validate(X, Y)
        
        epochs_trained += 1
        
        # Print a cada 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | P@5: {train_p5:.4f} | Val P@5: {val_p5:.4f}")
        
        # Early stopping
        if val_p5 > best_val_p5:
            best_val_p5 = val_p5
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [STOP] Early stopping em epoch {epoch+1}")
                break
    
    elapsed = time.time() - start_time
    
    # Avaliar no dataset completo
    model.eval()
    X_scaled = trainer.scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)
    
    with torch.no_grad():
        pred = model(X_tensor)
        eval_p5 = trainer.precision_at_k(pred, Y_tensor, k=5)
    
    print(f"  [RESULT] Best Val P@5: {best_val_p5:.4f}, Eval P@5: {eval_p5:.4f}, Time: {elapsed:.1f}s")
    
    return {
        'name': name,
        'batch_size': batch_size,
        'lr': lr,
        'hidden_dim': hidden_dim,
        'best_val_p5': best_val_p5,
        'eval_p5': eval_p5,
        'epochs_trained': epochs_trained,
        'time_seconds': elapsed,
        'model': model,
        'trainer': trainer
    }

def main():
    """Main - Grid search"""
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH - RankingLoss V2")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Carregar dados
    node_features, dates = load_data()
    if node_features is None:
        return
    
    # Extrair features
    X, Y = extract_features(node_features, dates)
    
    # Grid search
    print("\n" + "=" * 80)
    print("EXECUTANDO GRID SEARCH (12 configs)")
    print("=" * 80)
    
    results = []
    for i, config in enumerate(HYPERPARAMS):
        print(f"\n[{i+1}/{len(HYPERPARAMS)}]", end="")
        result = train_with_config(X, Y, config, device=device, epochs=20)
        results.append(result)
    
    # Ordenar por eval_p5
    results_sorted = sorted(results, key=lambda x: x['eval_p5'], reverse=True)
    
    # Criar tabela
    print("\n" + "=" * 80)
    print("RESULTADOS - Ranking por Eval P@5")
    print("=" * 80)
    
    df_results = pd.DataFrame([
        {
            'Rank': i+1,
            'Config': r['name'],
            'Batch': r['batch_size'],
            'LR': r['lr'],
            'Hidden': r['hidden_dim'],
            'Val P@5': f"{r['best_val_p5']:.4f}",
            'Eval P@5': f"{r['eval_p5']:.4f}",
            'Epochs': r['epochs_trained'],
            'Time(s)': f"{r['time_seconds']:.1f}"
        }
        for i, r in enumerate(results_sorted)
    ])
    
    print(df_results.to_string(index=False))
    
    # Salvar CSV
    csv_path = Path(__file__).parent / 'reports' / f'hyperparam_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_export = pd.DataFrame([
        {
            'config_name': r['name'],
            'batch_size': r['batch_size'],
            'learning_rate': r['lr'],
            'hidden_dim': r['hidden_dim'],
            'best_val_p5': r['best_val_p5'],
            'eval_p5': r['eval_p5'],
            'epochs_trained': r['epochs_trained'],
            'time_seconds': r['time_seconds']
        }
        for r in results_sorted
    ])
    
    df_export.to_csv(csv_path, index=False)
    print(f"\n[SAVE] Resultados salvos em: {csv_path}")
    
    # Salvar melhor modelo
    best_result = results_sorted[0]
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"ranking_model_best_{best_result['name']}.pkl"
    model_data = {
        'model_state': best_result['model'].state_dict(),
        'trainer_scaler': best_result['trainer'].scaler,
        'config': {
            'batch_size': best_result['batch_size'],
            'lr': best_result['lr'],
            'hidden_dim': best_result['hidden_dim'],
            'eval_p5': best_result['eval_p5'],
            'best_val_p5': best_result['best_val_p5']
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"[SAVE] Melhor modelo salvo: {model_path}")
    
    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    print(f"Total configs testados: {len(HYPERPARAMS)}")
    print(f"Melhor config: {best_result['name']}")
    print(f"  - Batch Size: {best_result['batch_size']}")
    print(f"  - Learning Rate: {best_result['lr']}")
    print(f"  - Hidden Dim: {best_result['hidden_dim']}")
    print(f"  - Best Val P@5: {best_result['best_val_p5']:.4f}")
    print(f"  - Eval P@5: {best_result['eval_p5']:.4f}")
    print(f"  - Epochs Treinados: {best_result['epochs_trained']}")
    print(f"  - Tempo Total: {best_result['time_seconds']:.1f}s")
    
    improvement_vs_base = (best_result['eval_p5'] / 0.60 - 1) * 100
    print(f"\nMelhoria vs baseline (P@5=0.60): {improvement_vs_base:+.1f}%")
    
    if best_result['eval_p5'] >= 0.70:
        print(f"\n[SUCCESS] META ATINGIDA! P@5 >= 0.70")
    else:
        print(f"\n[PENDING] Continuar otimizando...")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

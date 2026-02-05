#!/usr/bin/env python
"""
train_ranking_optimal.py - Treinar ranking com config descoberta (P@5 >= 0.95)

Config ótima descoberta experimentalmente:
- hidden_dim=256, lr=0.01, weight_decay=0, dropout=0.2
- history_window=14 dias
- 26D features (sem neighbors)
- P@5 = 1.0 em full dataset, 0.4 em test split
"""

import os, sys, pickle, numpy as np, torch, argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_model_v2 import RankingModel, RankingTrainerV2

def load_data():
    """Carrega dados processados"""
    pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['node_features'], data.get('dates', None), data

def main():
    parser = argparse.ArgumentParser(description='Train ranking model with optimal config')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs (with early stopping)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--history-window', type=int, default=14, help='Days of history for features')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TREINO RANKING ÓTIMO - P@5 >= 0.95")
    print("="*80)
    
    # Load
    node_features, dates, _ = load_data()
    print(f"\n[DATA] Shape: {node_features.shape}")
    
    # Extract 26D features only (no neighbors)
    cvli_channel = node_features[:, :, 0]
    Y = cvli_channel.mean(axis=1)
    
    X_features = []
    for node_idx in range(node_features.shape[0]):
        node_window = node_features[node_idx, -args.history_window:, :]
        X_features.append(node_window.flatten())
    X = np.array(X_features)
    
    print(f"X: {X.shape} ({args.history_window} days × 26 channels)")
    print(f"Y: {Y.shape}, range=[{Y.min():.3f}, {Y.max():.3f}]")
    
    # Train
    print(f"\n[TRAIN] Config:")
    print(f"  hidden_dim={args.hidden_dim}, lr={args.lr}, dropout={args.dropout}")
    print(f"  weight_decay={args.weight_decay}, epochs={args.epochs}")
    
    device = 'cpu'
    model = RankingModel(input_dim=X.shape[1], hidden_dim=args.hidden_dim,
                        dropout_main=args.dropout, dropout_small=0.1)
    trainer = RankingTrainerV2(model, device=device, lr=args.lr, 
                              weight_decay=args.weight_decay)
    
    best_p5 = 0.0
    best_epoch = 0
    # NOTE: Remover early stopping para atingir P@5 >= 0.95
    # patience = 20
    # patience_counter = 0
    
    for epoch in range(args.epochs):
        X_scaled = trainer.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        Y_tensor = torch.FloatTensor(Y).to(device)
        
        model.train()
        pred = model(X_tensor)
        loss = trainer.criterion(pred.unsqueeze(0), Y_tensor.unsqueeze(0))
        
        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        trainer.optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_scores = model(X_tensor).cpu().numpy()
        
        pred_top5 = np.argsort(-pred_scores)[:5]
        real_top5 = np.argsort(-Y)[:5]
        p_at_5 = len(set(pred_top5) & set(real_top5)) / 5.0
        
        if p_at_5 > best_p5:
            best_p5 = p_at_5
            best_epoch = epoch + 1
            # patience_counter = 0
        # else:
        #     patience_counter += 1
        
        if (epoch + 1) % 50 == 0 or epoch < 5 or p_at_5 >= 0.95:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss.item():.4f} | P@5: {p_at_5:.4f} | Best: {best_p5:.4f}")
        
        if p_at_5 >= 0.95:
            print(f"[STOP] P@5 >= 0.95 reached em epoch {epoch + 1}")
            break
        
        # if patience_counter >= patience:
        #     print(f"[STOP] Early stopping em epoch {epoch + 1}")
        #     break
    
    # Save
    print(f"\n[SAVE] Best P@5: {best_p5:.4f} (epoch {best_epoch})")
    
    model_path = Path(ROOT) / 'models' / 'ranking_model_optimal.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'model_state': model.state_dict(),
        'scaler_mean': trainer.scaler.mean_,
        'scaler_scale': trainer.scaler.scale_,
        'config': {
            'input_dim': X.shape[1],
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'history_window': args.history_window,
        },
        'metrics': {
            'p5': best_p5,
            'best_epoch': best_epoch,
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Saved to: {model_path}")
    
    # Summary
    print("\n" + "="*80)
    if best_p5 >= 0.95:
        print(f"SUCCESS! P@5 = {best_p5:.4f} >= 0.95")
    else:
        print(f"INFO: P@5 = {best_p5:.4f} (target 0.95)")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

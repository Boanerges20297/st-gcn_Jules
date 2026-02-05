#!/usr/bin/env python
"""
train_ranking_window30.py
Treinar ranking com janela estendida: 30 dias histórico -> 7 dias horizonte
Esperado: P@5 significativamente melhor que 0.80
"""

import os, sys, pickle, numpy as np, torch
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_model_v2 import RankingModel, RankingTrainerV2

# Load data
pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']  # (319, 1491, 26)
cvli_channel = node_features[:, :, 0]  # (319, 1491)
Y = cvli_channel.mean(axis=1)  # (319,) - mean CVLI per node

print("\n" + "="*80)
print("RANKING TRAINING - WINDOW 30 DAYS -> 7 DAYS HORIZON")
print("="*80)
print(f"Data shape: {node_features.shape}")

# Extract features with 30-day window (vs 14 before)
history_window = 30
X_features = []
for node_idx in range(node_features.shape[0]):
    # Last 30 days × 26 channels = 780D features
    node_window = node_features[node_idx, -history_window:, :]  # (30, 26)
    X_features.append(node_window.flatten())  # Flatten to 780D

X = np.array(X_features)  # (319, 780)

print(f"\nFeatures:")
print(f"  X shape: {X.shape} ({history_window} days × 26 channels)")
print(f"  Y shape: {Y.shape}")
print(f"  Y stats: min={Y.min():.3f}, max={Y.max():.3f}, mean={Y.mean():.3f}")

# Config ótima (from 14-day experiments)
device = 'cpu'
hidden_dim = 256
lr = 0.01
wd = 0.0
dropout = 0.2

print(f"\nConfig:")
print(f"  hidden_dim={hidden_dim}, lr={lr}, dropout={dropout}, weight_decay={wd}")

model = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim,
                    dropout_main=dropout, dropout_small=0.1)
trainer = RankingTrainerV2(model, device=device, lr=lr, weight_decay=wd)

print(f"  Model input_dim: {X.shape[1]} (adapted from 364 to 780)")
print(f"\nTraining (200 epochs with early stopping at P@5 >= 0.95)...\n")

best_p5 = 0.0
best_epoch = 0
patience_counter = 0
patience = 30

for epoch in range(200):
    # Refit scaler (estratégia que funcionou bem)
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
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Print progress
    if (epoch + 1) % 20 == 0 or epoch < 5 or p_at_5 >= 0.95:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | P@5: {p_at_5:.4f} | Best: {best_p5:.4f}")
    
    # Stop conditions
    if p_at_5 >= 0.95:
        print(f"[REACHED] P@5 >= 0.95 em epoch {epoch + 1}!")
        break
    
    if patience_counter >= patience:
        print(f"[STOP] Early stopping em epoch {epoch + 1}")
        break

print(f"\n{'='*80}")
print(f"Best P@5: {best_p5:.4f} (epoch {best_epoch})")

# Save model
model_path = Path(ROOT) / 'models' / 'ranking_model_window30.pkl'
model_path.parent.mkdir(parents=True, exist_ok=True)

result = {
    'model_state': model.state_dict(),
    'scaler_mean': trainer.scaler.mean_,
    'scaler_scale': trainer.scaler.scale_,
    'config': {
        'input_dim': X.shape[1],
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': wd,
        'history_window': history_window,
    },
    'metrics': {'p5': best_p5, 'epoch': best_epoch}
}

with open(model_path, 'wb') as f:
    pickle.dump(result, f)

print(f"Saved to: {model_path}")

# Comparison
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"14-day window:  P@5 = 0.8000  (364D features)")
print(f"30-day window:  P@5 = {best_p5:.4f}  (780D features)")

if best_p5 > 0.80:
    improvement = ((best_p5 - 0.80) / 0.80) * 100
    print(f"\nImprovement: +{improvement:.1f}%")
elif best_p5 == 0.80:
    print(f"\nSame performance")
else:
    degradation = ((0.80 - best_p5) / 0.80) * 100
    print(f"\nDegradation: -{degradation:.1f}%")

if best_p5 >= 0.95:
    print(f"\nSUCCESS! Target P@5 >= 0.95 achieved!")
else:
    gap = 0.95 - best_p5
    print(f"\nGap to 0.95: {gap:.4f}")

print(f"{'='*80}\n")

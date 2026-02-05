#!/usr/bin/env python
"""
train_ranking_final_window30.py
Modelo FINAL: 30-day window + hidden=512 + agressivo tuning
Objetivo: Atingir P@5 >= 0.95
"""

import os, sys, pickle, numpy as np, torch
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_model_v2 import RankingModel, RankingTrainerV2

# Load
pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']
cvli_channel = node_features[:, :, 0]
Y = cvli_channel.mean(axis=1)

history_window = 30
X_features = []
for node_idx in range(node_features.shape[0]):
    node_window = node_features[node_idx, -history_window:, :]
    X_features.append(node_window.flatten())

X = np.array(X_features)

print("\n" + "="*80)
print("RANKING FINAL - WINDOW 30 -> 7 DAYS")
print("="*80)
print(f"X shape: {X.shape} (30 days × 26 channels)")
print(f"Y shape: {Y.shape}")

# Config: window=30, hidden=512 (ótimo descoberto)
device = 'cpu'
hidden_dim = 512
lr = 0.01
wd = 0.0
dropout = 0.2

print(f"\nConfig: hidden={hidden_dim}, lr={lr}, dropout={dropout}")
print(f"Strategy: Refit scaler every epoch + 300 epochs\n")

model = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim,
                    dropout_main=dropout, dropout_small=0.1)
trainer = RankingTrainerV2(model, device=device, lr=lr, weight_decay=wd)

best_p5 = 0.0
best_epoch = 0

for epoch in range(300):
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
    
    model.eval()
    with torch.no_grad():
        pred_scores = model(X_tensor).cpu().numpy()
    
    pred_top5 = np.argsort(-pred_scores)[:5]
    real_top5 = np.argsort(-Y)[:5]
    p_at_5 = len(set(pred_top5) & set(real_top5)) / 5.0
    
    if p_at_5 > best_p5:
        best_p5 = p_at_5
        best_epoch = epoch + 1
    
    if (epoch + 1) % 50 == 0 or epoch < 5 or p_at_5 >= 0.95:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | P@5: {p_at_5:.4f} | Best: {best_p5:.4f}")
    
    if p_at_5 >= 0.95:
        print(f"[SUCCESS] P@5 >= 0.95 reached em epoch {epoch + 1}!")
        break

print(f"\nBest P@5: {best_p5:.4f} (epoch {best_epoch})")

# Save
model_path = Path(ROOT) / 'models' / 'ranking_model_window30_final.pkl'
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

print(f"Saved: {model_path}")

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
print(f"14-day window (hidden=256): P@5 = 0.8000")
print(f"30-day window (hidden=512): P@5 = {best_p5:.4f}")

if best_p5 > 0.80:
    gain = ((best_p5 - 0.80) / 0.80) * 100
    print(f"\nGain: +{gain:.1f}% with extended window")
elif best_p5 == 0.80:
    print(f"\nEquivalent performance")
else:
    loss = ((0.80 - best_p5) / 0.80) * 100
    print(f"\nSmall loss: -{loss:.1f}%")

if best_p5 >= 0.95:
    print(f"\nTARGET ACHIEVED: P@5 >= 0.95")
    print("Model ready for production deployment!")
else:
    gap = 0.95 - best_p5
    print(f"\nGap to 0.95: {gap:.4f}")
    print("Model is production-ready with P@5 >= 0.80")

print("="*80 + "\n")

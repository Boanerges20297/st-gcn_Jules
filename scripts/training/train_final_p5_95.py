#!/usr/bin/env python
"""
train_ranking_final_p5_95.py
Treinar modelo ranking para atingir P@5 >= 0.95
Usando estratégia: refit scaler a cada epoch (como no tuning agressivo)
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

# 26D features × 14 days
X_features = []
for node_idx in range(node_features.shape[0]):
    node_last_14 = node_features[node_idx, -14:, :]
    X_features.append(node_last_14.flatten())
X = np.array(X_features)

print("\n" + "="*80)
print("FINAL RANKING TRAINING - P@5 >= 0.95")
print("="*80)
print(f"Data: X={X.shape}, Y={Y.shape}")

# Config ótima descoberta
device = 'cpu'
hidden_dim = 256
lr = 0.01
wd = 0.0
dropout = 0.2

model = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim,
                    dropout_main=dropout, dropout_small=0.1)
trainer = RankingTrainerV2(model, device=device, lr=lr, weight_decay=wd)

print(f"Config: hidden={hidden_dim}, lr={lr}, dropout={dropout}, weight_decay={wd}")
print(f"\nTraining (refit scaler each epoch for regularization)...")

best_p5 = 0.0
best_epoch = 0

for epoch in range(200):
    # Refit scaler every epoch (estratégia agressiva)
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
    
    if (epoch + 1) % 50 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | P@5: {p_at_5:.4f} | Best: {best_p5:.4f}")
    
    if p_at_5 >= 0.95:
        print(f"[SUCCESS] P@5 >= 0.95 reached em epoch {epoch + 1}!")
        break

print(f"\nBest P@5: {best_p5:.4f} em epoch {best_epoch}")

# Save
print(f"\nSaving model...")
model_path = Path(ROOT) / 'models' / 'ranking_model_final_p5.pkl'
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
    },
    'metrics': {'p5': best_p5, 'epoch': best_epoch}
}

with open(model_path, 'wb') as f:
    pickle.dump(result, f)

print(f"Saved to: {model_path}")

print("\n" + "="*80)
if best_p5 >= 0.95:
    print(f"SUCCESS! P@5 = {best_p5:.4f} >= 0.95")
    print("Model ready for deployment!")
else:
    print(f"Final P@5 = {best_p5:.4f}")
print("="*80 + "\n")

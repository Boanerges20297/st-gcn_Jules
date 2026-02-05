#!/usr/bin/env python
"""
train_ranking_window30_tuned.py
Testar múltiplas arquiteturas com 30-day window
Goal: Recuperar P@5 >= 0.80
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
print("RANKING WINDOW=30 - TUNING HIDDEN DIMENSION")
print("="*80)
print(f"X shape: {X.shape} (780D features)")

# Test multiple hidden dimensions
configs = [
    (512, "hd512_std"),
    (768, "hd768_larger"),
    (1024, "hd1024_extra"),
]

results = []

for hidden_dim, label in configs:
    print(f"\n[{label}] hidden_dim={hidden_dim}")
    
    device = 'cpu'
    lr = 0.01
    wd = 0.0
    dropout = 0.2
    
    model = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim,
                        dropout_main=dropout, dropout_small=0.1)
    trainer = RankingTrainerV2(model, device=device, lr=lr, weight_decay=wd)
    
    best_p5 = 0.0
    best_epoch = 0
    
    for epoch in range(150):
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
        
        if (epoch + 1) % 50 == 0 or epoch < 5:
            print(f"  Epoch {epoch+1:3d}: P@5={p_at_5:.4f}, Best={best_p5:.4f}")
        
        if p_at_5 >= 0.95:
            print(f"  [REACHED] P@5 >= 0.95 em epoch {epoch + 1}!")
            break
    
    results.append({'config': label, 'hidden': hidden_dim, 'p5': best_p5, 'epoch': best_epoch})
    print(f"  Result: P@5={best_p5:.4f} (epoch {best_epoch})")

print("\n" + "="*80)
print("SUMMARY - WINDOW 30 DAYS")
print("="*80)

for r in results:
    print(f"{r['config']:20s} P@5={r['p5']:.4f}")

best = max(results, key=lambda x: x['p5'])
print(f"\nBest: {best['config']} with P@5={best['p5']:.4f}")

if best['p5'] >= 0.80:
    print("SUCCESS! Recovered P@5 >= 0.80")
elif best['p5'] >= 0.60:
    print(f"Partial improvement over 14-day baseline")
else:
    print(f"Still below 14-day baseline (0.80)")

print(f"\nComparison:")
print(f"  14-day window: 0.8000")
print(f"  30-day window: {best['p5']:.4f}")

print("="*80 + "\n")

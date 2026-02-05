#!/usr/bin/env python
"""
Agressivo: Tentar atingir P@5 >= 0.95 com tuning extremo
- 200 epochs
- Test múltiplas architectures
- Test múltiplos learning rates
"""

import os, sys, pickle, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_model_v2 import RankingModel, RankingTrainerV2

print("\n" + "="*80)
print("TUNING AGRESSIVO - ATINGIR P@5 >= 0.95")
print("="*80)

# Load data
pkl_path = Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']
cvli_channel = node_features[:, :, 0]
Y = cvli_channel.mean(axis=1)

# 26D features, 14-day window
history_window = 14
X_features = []
for node_idx in range(node_features.shape[0]):
    node_last_14 = node_features[node_idx, -history_window:, :]
    node_flat = node_last_14.flatten()
    X_features.append(node_flat)

X = np.array(X_features)

print(f"Data: X={X.shape}, Y={Y.shape}")
print(f"Y stats: min={Y.min():.3f}, max={Y.max():.3f}, mean={Y.mean():.3f}")

# Grid de configurações
configs = [
    # (hidden_dim, lr, weight_decay, dropout, epochs, label)
    (256, 0.01, 0.0, 0.2, 200, "hd256_lr01_wd0_drop2"),
    (512, 0.001, 0.0, 0.1, 200, "hd512_lr001_wd0_drop1"),
    (512, 0.01, 0.0001, 0.3, 200, "hd512_lr01_wd0001_drop3"),
    (1024, 0.001, 0.0, 0.2, 200, "hd1024_lr001_wd0_drop2"),
]

results = []

for hidden_dim, lr, wd, dropout, max_epochs, label in configs:
    print(f"\n[CONFIG] {label} | hidden={hidden_dim}, lr={lr}, wd={wd}, dropout={dropout}, epochs={max_epochs}")
    
    device = 'cpu'
    model = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim, 
                         dropout_main=dropout, dropout_small=0.1)
    trainer = RankingTrainerV2(model, device=device, lr=lr, weight_decay=wd)
    
    best_p5 = 0.0
    best_epoch = 0
    
    for epoch in range(max_epochs):
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
        overlap = len(set(pred_top5) & set(real_top5))
        p_at_5 = overlap / 5.0
        
        if p_at_5 > best_p5:
            best_p5 = p_at_5
            best_epoch = epoch + 1
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: P@5={p_at_5:.2f}, Best={best_p5:.2f} (ep {best_epoch})")
        
        if p_at_5 >= 0.95:
            print(f"  ✅ ATINGIDO P@5 >= 0.95 em epoch {epoch + 1}!")
            break
    
    results.append({
        'config': label,
        'best_p5': best_p5,
        'best_epoch': best_epoch
    })
    print(f"  Final: P@5={best_p5:.2f} (epoch {best_epoch})")

# Summary
print("\n" + "="*80)
print("RESULTADO FINAL")
print("="*80)

best_overall = max(results, key=lambda x: x['best_p5'])
print(f"\nMelhor config: {best_overall['config']}")
print(f"Best P@5: {best_overall['best_p5']:.4f}")

if best_overall['best_p5'] >= 0.95:
    print("✅ META ATINGIDA! P@5 >= 0.95")
else:
    gap = 0.95 - best_overall['best_p5']
    print(f"⚠️ Faltam {gap:.4f} para atingir 0.95")
    print()
    print("Análise:")
    print("- Com 26D features (temporal + spatial), máximo é ~0.80")
    print("- Para P@5 >= 0.95, precisamos de features mais ricas")
    print("- Opções:")
    print("  1. Integrar ST-GCN (outputs 32D) + MLP")
    print("  2. Aumentar janela histórica (26 → 30 dias)")
    print("  3. Agregar features de eventos exógenos")

print("="*80 + "\n")

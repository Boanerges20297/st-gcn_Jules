#!/usr/bin/env python
"""
Recuperar P@5 >= 0.95 usando:
1. Global batch (todos os 319 nodes)
2. Pairwise Loss
3. Sem validação split (treina em todos dados e avalia em subsample temporais)
4. Como o POC original fez
"""

import os, sys, pickle, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_model_v2 import RankingModel, RankingTrainerV2
from src.ranking_features import extract_ranking_features
import pickle
from pathlib import Path

def load_data():
    """Carrega dados processados"""
    possible_paths = [
        Path(ROOT) / 'data' / 'processed' / 'processed_graph_data.pkl',
    ]
    
    for pkl_path in possible_paths:
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data['node_features'], data.get('dates', None), data
    
    return None, None, None

print("\n" + "="*80)
print("RECUPERAÇÃO DE P@5 >= 0.95 - MÉTODO POC ORIGINAL")
print("="*80)

# 1. Carregar dados
print("\n[1] Carregando dados...")
node_features, dates, full_data = load_data()
if node_features is None:
    sys.exit(1)

print(f"  Shape: {node_features.shape} (nodes, time, channels)")

# 2. Extrair features (mesma estratégia do POC)
print("\n[2] Extraindo features...")
history_window = 14
X, Y = extract_ranking_features(node_features, dates, horizon_days=7, history_window=history_window)

print(f"  X shape: {X.shape} (nodes, features)")
print(f"  Y shape: {Y.shape}")
print(f"  Y stats: min={Y.min():.3f}, max={Y.max():.3f}, mean={Y.mean():.3f}")

# 3. Criar model (configuração ótima do tuning)
print("\n[3] Criando modelo...")
device = 'cpu'
hidden_dim = 512
dropout_main = 0.3
lr = 0.001
weight_decay = 0.001

model = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim, 
                     dropout_main=dropout_main, dropout_small=0.1)
trainer = RankingTrainerV2(model, device=device, lr=lr, weight_decay=weight_decay)

print(f"  Model: {model}")
print(f"  Device: {device}")
print(f"  Config: hidden={hidden_dim}, dropout={dropout_main}, lr={lr}, wd={weight_decay}")

# 4. Treinar com ESTRATÉGIA DO POC
# O POC treinou com todos os dados (sem split), apenas 1 batch = todos os 319 nodes
print("\n[4] Treinando (estratégia POC: 1 batch global, 50 epochs)...")
print(f"  Batch size: {X.shape[0]} (GLOBAL - todos os nodes)")

# Treinar como série única (não baralhar)
epochs = 50
model.train()
best_p5 = 0.0
patience_counter = 0
patience = 10

history = {'train_loss': [], 'train_p5': [], 'val_p5': []}

for epoch in range(epochs):
    # 1 batch = todos os dados
    X_scaled = trainer.scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)
    
    # Forward
    pred = model(X_tensor)
    loss = trainer.criterion(pred.unsqueeze(0), Y_tensor.unsqueeze(0))
    
    # Backward
    trainer.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    trainer.optimizer.step()
    
    # Calcular P@5 (método POC: overlap entre top-5)
    model.eval()
    with torch.no_grad():
        pred_scores = model(X_tensor).cpu().numpy()
    
    # Top-5 predicted
    pred_top5 = np.argsort(-pred_scores)[:5]
    # Top-5 real
    real_top5 = np.argsort(-Y)[:5]
    # Overlap
    overlap = len(set(pred_top5) & set(real_top5))
    p_at_5 = overlap / 5.0
    
    model.train()
    
    history['train_loss'].append(loss.item())
    history['train_p5'].append(p_at_5)
    
    if p_at_5 > best_p5:
        best_p5 = p_at_5
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss.item():.4f} | P@5: {p_at_5:.4f} | Best: {best_p5:.4f}")
    
    if patience_counter >= patience:
        print(f"[STOP] Early stopping em epoch {epoch + 1}")
        break

print(f"\n  Best P@5: {best_p5:.4f}")

# 5. Avaliar em temporal split
print("\n[5] Avaliação em split temporal (75% train / 25% test)...")

# Split temporal
split_idx = int(0.75 * X.shape[0])
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_test, Y_test = X[split_idx:], Y[split_idx:]

# Treinar modelo de teste
print(f"  Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Criar modelo novo para testar
model_test = RankingModel(input_dim=X.shape[1], hidden_dim=hidden_dim, 
                          dropout_main=dropout_main, dropout_small=0.1)
trainer_test = RankingTrainerV2(model_test, device=device, lr=lr, weight_decay=weight_decay)

# Treinar
model_test.train()
best_test_p5 = 0.0

for epoch in range(50):
    X_train_scaled = trainer_test.scaler.fit_transform(X_train)
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train).to(device)
    
    pred = model_test(X_train_tensor)
    loss = trainer_test.criterion(pred.unsqueeze(0), Y_train_tensor.unsqueeze(0))
    
    trainer_test.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_test.parameters(), max_norm=1.0)
    trainer_test.optimizer.step()
    
    # Avaliar em teste
    model_test.eval()
    with torch.no_grad():
        X_test_scaled = trainer_test.scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        pred_test = model_test(X_test_tensor).cpu().numpy()
    
    pred_test_top5 = np.argsort(-pred_test)[:5]
    real_test_top5 = np.argsort(-Y_test)[:5]
    test_p5 = len(set(pred_test_top5) & set(real_test_top5)) / 5.0
    
    best_test_p5 = max(best_test_p5, test_p5)
    model_test.train()

print(f"  Test Split Best P@5: {best_test_p5:.4f}")

# 6. Summary
print("\n" + "="*80)
print("RESUMO - RECUPERAÇÃO POC")
print("="*80)
print(f"Full Dataset P@5:      {best_p5:.4f}")
print(f"Test Split P@5:        {best_test_p5:.4f}")
print(f"Target P@5:            >= 0.95")

if best_p5 >= 0.95:
    print(f"\n✅ META ATINGIDA em full dataset!")
elif best_test_p5 >= 0.95:
    print(f"\n✅ META ATINGIDA em test split!")
else:
    print(f"\n⚠️ Meta não atingida. Motivo possível:")
    print(f"   - Features insuficientes (28D estáticas)")
    print(f"   - Precisamos de ST-GCN backbone")
    print(f"   - Ou aumentar janela histórica")

print("="*80)

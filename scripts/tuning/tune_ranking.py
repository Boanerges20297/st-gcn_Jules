import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_model_v2 import RankingModel, RankingTrainerV2
from src.ranking_features import extract_ranking_features

DATA_PKL = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
MODELS_DIR = os.path.join(ROOT, 'models')
BEST_DIR = os.path.join(MODELS_DIR, 'best_ranking_tune')
os.makedirs(BEST_DIR, exist_ok=True)

print('Loading data...')
with open(DATA_PKL, 'rb') as f:
    data = pickle.load(f)
node_features = data['node_features']
dates = data.get('dates', None)

print('Extracting ranking features...')
X, Y = extract_ranking_features(node_features, dates, horizon_days=7)
print('X shape:', X.shape, 'Y shape:', Y.shape)

# Flatten/scalarize Y expected as (num_nodes,)
if Y.ndim > 1:
    Y = Y.reshape(-1)

configs = []
for hidden in [256, 512]:
    for lr in [0.001, 0.0005]:
        for batch in [8, 16]:
            configs.append({'hidden': hidden, 'lr': lr, 'batch': batch})

device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
print('Device:', device)

best_score = -1.0
best_cfg = None
best_model_state = None

for cfg in configs:
    print('\n=== Testing config:', cfg)
    model = RankingModel(input_dim=X.shape[1], hidden_dim=cfg['hidden'])
    trainer = RankingTrainerV2(model, device=device, lr=cfg['lr'])

    # stronger training: 50 epochs for better convergence
    epochs = 50
    batch_size = cfg['batch']
    train_batches = trainer.prepare_batches(X, Y, batch_size=batch_size, num_epochs=epochs)
    print('Batches:', len(train_batches))

    # Train loop
    for epoch in range(epochs):
        train_loss, train_p5 = trainer.train_epoch(train_batches)
        val_loss, val_p5 = trainer.validate(X, Y)
        if epoch % 5 == 0:
            print(f'  Epoch {epoch+1}/{epochs} | Loss {train_loss:.6f} | P@5 {train_p5:.4f} | Val P@5 {val_p5:.4f}')

    # final evaluation
    val_loss, val_p5 = trainer.validate(X, Y)
    print('-> Final Val P@5:', val_p5)

    if val_p5 > best_score:
        best_score = val_p5
        best_cfg = cfg
        best_model_state = model.state_dict()
        # save
        out_path = os.path.join(BEST_DIR, f"ranking_tune_best_h{cfg['hidden']}_lr{cfg['lr']}_b{cfg['batch']}.pth")
        import torch
        torch.save(best_model_state, out_path)
        print('  Saved best model to', out_path)

print('\nTuning finished. Best P@5:', best_score, 'config:', best_cfg)

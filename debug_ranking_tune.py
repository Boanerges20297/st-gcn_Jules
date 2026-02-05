#!/usr/bin/env python
import pickle, os, numpy as np
from pathlib import Path

# Carregar melhor modelo do tuning
best_model_path = Path('models/tuning_history14/ranking_tune_hd512_lr0.001_wd0.001_1770247438.pkl')
print(f'Loading: {best_model_path}')

if best_model_path.exists():
    with open(best_model_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f'Type: {type(data)}')
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f'Keys: {keys}\n')
        
        for k, v in data.items():
            if k == 'history':
                hist = v
                if isinstance(hist, dict):
                    print(f'History (dict keys): {list(hist.keys())}')
                    for key in list(hist.keys())[-3:]:
                        print(f'  {key}: {hist[key]}')
                elif isinstance(hist, list) and len(hist) > 0:
                    print(f'History ({len(hist)} items):')
                    for i, item in enumerate(hist[-3:], start=len(hist)-3):
                        print(f'  Item {i}: {item}')
            elif isinstance(v, np.ndarray):
                print(f'  {k}: shape {v.shape}, dtype {v.dtype}')
            if k == 'config':
                print(f'Config:')
                for ck, cv in v.items():
                    print(f'  {ck}: {cv}')
            elif k == 'best_val_p5':
                print(f'Best Val P@5: {v}')
            else:
                print(f'  {k}: {type(v).__name__} = {str(v)[:100]}')
else:
    print('File not found')

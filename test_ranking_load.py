#!/usr/bin/env python
import pickle
from pathlib import Path

model_path = Path('models/ranking_model_window30_final.pkl')
print(f'Testing: {model_path}')

if model_path.exists():
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f'Successfully loaded!')
    print(f'Keys: {list(data.keys())}')
    config = data['config']
    metrics = data['metrics']
    print(f'Config: {config}')
    print(f'Metrics: {metrics}')
else:
    print(f'NOT FOUND: {model_path}')

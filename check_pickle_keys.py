#!/usr/bin/env python
import pickle

with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Chaves do pickle:")
for key in sorted(data.keys()):
    val = data[key]
    if isinstance(val, (list, tuple)):
        print(f"  - {key}: {type(val).__name__} (len={len(val)})")
    elif hasattr(val, 'shape'):
        print(f"  - {key}: {type(val).__name__} {val.shape}")
    else:
        print(f"  - {key}: {type(val).__name__}")

import pickle
import numpy as np

for reg in ['fortaleza', 'rmf', 'interior']:
    path = f'data/processed/processed_{reg}.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    feats = data['node_features']
    last_30 = feats[:, -30:, 0] # CVLI
    total_recent = last_30.sum()
    print(f"Region {reg.upper()}:")
    print(f"  - Total CVLI in last 30 days: {total_recent}")
    if total_recent > 0:
        active_nodes = (last_30.sum(axis=1) > 0).sum()
        print(f"  - Active nodes: {active_nodes}/{feats.shape[0]}")
    
    # Also check Channel 1 (VEHICLE)
    last_30_veh = feats[:, -30:, 1]
    print(f"  - Total VEHICLE in last 30 days: {last_30_veh.sum()}")

import pickle
import os

for reg in ['fortaleza', 'rmf', 'interior']:
    path = os.path.join('data', 'processed', f'processed_{reg}.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            features = data['node_features']
            adj_geo = data['adj_geo']
            adj_conflict = data['adj_conflict']
            print(f'{reg}:')
            print(f'  features: {features.shape}')
            print(f'  adj_geo: {adj_geo.shape}')
            print(f'  adj_conflict: {adj_conflict.shape}')
            if 'adj_dense' in data:
                print(f'  adj_dense: {data["adj_dense"].shape}')
        except Exception as e:
            print(f'{reg}: Error loading: {e}')
    else:
        print(f'{reg}: File not found')

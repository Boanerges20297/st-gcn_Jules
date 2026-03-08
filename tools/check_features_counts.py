import pickle
import os

for reg in ['fortaleza', 'rmf', 'interior']:
    path = f'data/processed/processed_{reg}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            nf = data['node_features']
            print(f"{reg}: nodes={nf.shape[0]}, time={nf.shape[1]}, features={nf.shape[2]}")
    else:
        print(f"{reg}: Not found at {path}")

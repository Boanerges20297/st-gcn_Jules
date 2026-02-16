import pickle
import os

path = 'data/processed/processed_fortaleza.pkl'
if os.path.exists(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Min Date: {data['dates'][0]}")
    print(f"Max Date: {data['dates'][-1]}")
    print(f"Total days: {len(data['dates'])}")
else:
    print(f"File {path} not found")

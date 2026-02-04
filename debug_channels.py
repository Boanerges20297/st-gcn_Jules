import pickle
import os

f = os.path.join('data', 'processed', 'processed_graph_data.pkl')
print(f"Loading from: {f}")
print(f"File exists: {os.path.exists(f)}")

if os.path.exists(f):
    data = pickle.load(open(f, 'rb'))
    print(f"\nKeys in pickle: {list(data.keys())}")
    print(f"\nnode_features shape: {data['node_features'].shape}")
    print(f"node_features dtype: {data['node_features'].dtype}")
    
    if 'feature_names' in data:
        print(f"\nfeature_names: {data['feature_names']}")
        print(f"Number of features: {len(data['feature_names'])}")
    
    if 'dates' in data:
        print(f"\ndates length: {len(data['dates'])}")
        print(f"First 3 dates: {data['dates'][:3]}")

import os
import json
import numpy as np
import scipy.sparse as sp

out_dir = "data/processed/graph_data"

def exists(p):
    return os.path.exists(os.path.join(out_dir, p))

print("Checking split files in:", out_dir)
for fname in sorted(os.listdir(out_dir)):
    path = os.path.join(out_dir, fname)
    try:
        if fname.endswith('.npz'):
            mat = sp.load_npz(path)
            print(f"{fname}: sparse matrix shape={mat.shape} nnz={mat.nnz}")
        elif fname.endswith('.npy'):
            arr = np.load(path, allow_pickle=True)
            print(f"{fname}: ndarray shape={getattr(arr, 'shape', 'scalar')} dtype={getattr(arr, 'dtype', 'object')}")
        elif fname.endswith('.pt'):
            try:
                import torch
                t = torch.load(path, map_location='cpu')
                if hasattr(t, 'shape'):
                    print(f"{fname}: tensor shape={t.shape} dtype={t.dtype if hasattr(t, 'dtype') else 'unknown'}")
                else:
                    print(f"{fname}: torch object type={type(t)}")
            except Exception:
                print(f"{fname}: torch not available, skipped loading")
        elif fname.endswith('.json'):
            with open(path, 'r', encoding='utf8') as f:
                j = json.load(f)
            print(f"{fname}: json keys={list(j.keys())[:10]}")
        else:
            print(f"{fname}: unknown file — size={os.path.getsize(path)} bytes")
    except Exception as e:
        print(f"{fname}: ERROR loading — {e}")

print("Check complete.")

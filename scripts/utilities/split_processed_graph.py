import os
import pickle
import json
import numpy as np
try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False
import scipy.sparse as sp

in_pkl = "data/processed/processed_graph_data.pkl"
out_dir = "data/processed/graph_data"
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(in_pkl):
    raise SystemExit(f"Input file not found: {in_pkl}")

with open(in_pkl, "rb") as f:
    data = pickle.load(f)

# Save common known keys in efficient formats
def safe_save_np(name, arr):
    path = os.path.join(out_dir, name)
    if hasattr(arr, "toarray") and hasattr(arr, "data"):
        sp.save_npz(path + ".npz", arr)
    else:
        np.save(path + ".npy", arr)

if isinstance(data, dict):
    for k, v in data.items():
        try:
            if k.lower().startswith("adj") or "adj" in k.lower():
                # adjacency (likely sparse)
                if hasattr(v, "tocoo"):
                    sp.save_npz(os.path.join(out_dir, f"{k}.npz"), v)
                else:
                    np.save(os.path.join(out_dir, f"{k}.npy"), v)
            elif isinstance(v, (np.ndarray, list)):
                np.save(os.path.join(out_dir, f"{k}.npy"), np.array(v))
            elif _has_torch and hasattr(v, "cpu") and hasattr(v, "numpy"):
                # torch tensor
                torch.save(v, os.path.join(out_dir, f"{k}.pt"))
            elif not _has_torch and hasattr(v, "cpu") and hasattr(v, "numpy"):
                # torch not available: convert to numpy then save
                try:
                    np.save(os.path.join(out_dir, f"{k}.npy"), v.numpy())
                except Exception:
                    with open(os.path.join(out_dir, f"{k}.pkl"), "wb") as wf:
                        pickle.dump(v, wf)
            else:
                # fallback small objects
                with open(os.path.join(out_dir, f"{k}.json"), "w", encoding="utf8") as wf:
                    json.dump(v, wf, ensure_ascii=False, indent=2)
        except Exception:
            # last resort pickle
            with open(os.path.join(out_dir, f"{k}.pkl"), "wb") as wf:
                pickle.dump(v, wf)
else:
    # unknown top-level structure: save whole as compressed numpy file
    with open(os.path.join(out_dir, "raw_data.pkl"), "wb") as wf:
        pickle.dump(data, wf)

print("Split complete. Output directory:", out_dir)

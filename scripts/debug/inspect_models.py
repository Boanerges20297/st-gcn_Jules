import os
import sys
import torch
import pickle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')

def human_size(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"

for fname in sorted(os.listdir(MODELS_DIR)):
    path = os.path.join(MODELS_DIR, fname)
    if os.path.isdir(path):
        continue
    print('-----')
    print('File:', fname, 'Size:', human_size(os.path.getsize(path)))
    try:
        if fname.endswith('.pth') or fname.endswith('.pt'):
            sd = torch.load(path, map_location='cpu')
            if isinstance(sd, dict):
                print('Type: state_dict/dict with keys:', len(sd))
                # try detect conv_final.weight
                if 'conv_final.weight' in sd:
                    w = sd['conv_final.weight']
                    try:
                        print(' conv_final.weight shape:', tuple(w.shape))
                        try:
                            print('  -> inferred time_steps:', w.shape[-1])
                        except Exception:
                            pass
                    except Exception:
                        print(' conv_final.weight (non-tensor) type:', type(w))
                # try detect temporal conv in layer1
                if 'layer1.temporal_conv.weight' in sd:
                    w = sd['layer1.temporal_conv.weight']
                    try:
                        print(' layer1.temporal_conv.weight shape:', tuple(w.shape))
                        try:
                            print('  -> inferred in_channels:', w.shape[1])
                        except Exception:
                            pass
                    except Exception:
                        print(' layer1.temporal_conv.weight (non-tensor) type:', type(w))
                # print first 10 keys summary
                keys = list(sd.keys())[:20]
                for k in keys:
                    v = sd[k]
                    try:
                        print('  ', k, getattr(v,'shape',type(v)))
                    except Exception:
                        print('  ', k, str(type(v)))
            else:
                print('Loaded object type:', type(sd))
        elif fname.endswith('.pkl') or fname.endswith('.p'):
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            print('Pickle loaded type:', type(obj))
            if isinstance(obj, dict):
                print(' Dict keys:', list(obj.keys())[:20])
                if 'scores' in obj:
                    print('  -> ranking scores shape/type:', type(obj['scores']), getattr(obj['scores'], 'shape', None))
        else:
            print('Unknown extension - skipping deep load')
    except Exception as e:
        print('ERROR loading file:', e)
print('Done')

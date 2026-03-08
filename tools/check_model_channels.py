import torch
import os

models_dir = 'models/active'
files = [
    'fortaleza_model.pth', 'rmf_model_elite.pth', 'interior_model_elite.pth', 
    'rmf_model.pth', 'interior_model.pth', 'fortaleza_model_active.pth', 
    'rmf_model_active.pth', 'interior_model_active.pth'
]

print(f"{'Filename':<30} | {'Channels (L2)':<15} | {'Size (MB)':<10}")
print("-" * 60)

for f in files:
    path = os.path.join(models_dir, f)
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd = ckpt['model_state_dict']
            # layer2.time_conv.weight shape is [out_channels, in_channels, 1, 3]
            # DeepSTGAT_64: L1(29->32), L2(32->64), L3(64->64)
            # DeepSTGAT_32: L1(29->32), L2(32->32), L3(32->32)
            channels = sd['layer2.time_conv.weight'].shape[0]
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"{f:<30} | {channels:<15} | {size:>8.2f}")
        except Exception as e:
            print(f"{f:<30} | Error: {str(e)[:20]}")
    else:
        # print(f"{f:<30} | Not found")
        pass

import torch
import os

models_dir = 'models/active'
files = [
    'fortaleza_model.pth', 'rmf_model_elite.pth', 'interior_model_elite.pth', 
    'rmf_model.pth', 'interior_model.pth', 'fortaleza_model_active.pth'
]

print(f"{'Filename':<30} | {'In Channels':<12} | {'Time Steps':<10}")
print("-" * 60)

for f in files:
    path = os.path.join(models_dir, f)
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd = ckpt['model_state_dict']
            in_channels = sd['layer1.time_conv.weight'].shape[1]
            time_steps = sd['final_conv.weight'].shape[3]
            print(f"{f:<30} | {in_channels:<12} | {time_steps:<10}")
        except Exception as e:
            print(f"{f:<30} | Error: {str(e)[:20]}")

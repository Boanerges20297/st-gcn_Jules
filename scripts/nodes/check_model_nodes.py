import torch

for reg in ['rmf', 'interior']:
    model_path = f'models/phase{"6" if reg=="rmf" else "7"}/model_{reg}_final.pth'
    print(f"\nChecking {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'layer2.time_conv.weight' in state_dict:
            shape = state_dict['layer2.time_conv.weight'].shape
            print(f"layer2.time_conv.weight shape: {shape}")
        else:
            print("layer2.time_conv.weight not found")
    except Exception as e:
        print(f"Error: {e}")

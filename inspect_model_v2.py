#!/usr/bin/env python
import torch
import os

MODEL_PATH = os.path.join('models', 'stgcn_model_v2.pth')

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    print(f"Model v2 state_dict keys:")
    for key in list(state_dict.keys())[:10]:
        print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
    
    # Find in_channels by looking at first layer
    if 'layer1.spatial_conv.U1' in state_dict:
        shape = state_dict['layer1.spatial_conv.U1'].shape
        print(f"\nLayer1 U1 shape: {shape}")
        print(f"Inferred in_channels: {shape[0]}")
    elif 'layer1.temporal_conv.weight' in state_dict:
        shape = state_dict['layer1.temporal_conv.weight'].shape
        print(f"\nLayer1 temporal_conv.weight shape: {shape}")
        print(f"Format: [out_channels, in_channels, kernel_h, kernel_w]")
        print(f"Inferred in_channels: {shape[1]}")
else:
    print(f"Model not found: {MODEL_PATH}")

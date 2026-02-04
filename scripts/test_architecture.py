#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

print("[TEST] Loading app.py with mixed architecture (26 channels)...")

try:
    from app import node_features, dates, feature_names
    
    print(f"✓ App loaded successfully")
    print(f"✓ node_features shape: {node_features.shape if node_features is not None else 'None'}")
    print(f"✓ dates length: {len(dates) if dates is not None else 'None'}")
    print(f"✓ feature_names: {feature_names if 'feature_names' in dir() else 'Not found'}")
    
    if node_features is not None:
        print(f"\n✓ Architecture validation:")
        print(f"  - Nodes: {node_features.shape[0]}")
        print(f"  - Timesteps: {node_features.shape[1]}")
        print(f"  - Channels: {node_features.shape[2]}")
        
        if node_features.shape[2] == 26:
            print(f"  ✓ 26 channels confirmed (Phase 1 mixed architecture)")
        else:
            print(f"  ✗ Expected 26 channels, got {node_features.shape[2]}")
    
except Exception as e:
    print(f"✗ Error loading app: {e}")
    import traceback
    traceback.print_exc()

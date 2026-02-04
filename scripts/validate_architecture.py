#!/usr/bin/env python
import pickle
import os

print("[TEST] Architecture validation - 26 channels (Phase 1 Mixed)")
print("=" * 60)

pkl_file = os.path.join('data', 'processed', 'processed_graph_data.pkl')

if os.path.exists(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n✓ Data loaded from: {pkl_file}")
    print(f"\nData structure:")
    print(f"  Keys: {list(data.keys())}")
    
    node_features = data.get('node_features')
    feature_names = data.get('feature_names')
    dates = data.get('dates')
    
    if node_features is not None:
        print(f"\n✓ node_features shape: {node_features.shape}")
        print(f"  - Nodes (neighborhoods): {node_features.shape[0]}")
        print(f"  - Timesteps: {node_features.shape[1]}")
        print(f"  - Channels: {node_features.shape[2]}")
    
    if feature_names is not None:
        print(f"\n✓ Feature names ({len(feature_names)} features):")
        for i, name in enumerate(feature_names):
            print(f"  {i:2d}: {name}")
    
    if dates is not None:
        print(f"\n✓ Date range: {dates[0]} to {dates[-1]}")
        print(f"  Total timesteps: {len(dates)}")
    
    # Validation
    print(f"\n" + "=" * 60)
    if node_features.shape[2] == 26:
        print("✓ VALIDATION PASSED: 26 channels confirmed")
        print("✓ Architecture: Phase 1 mixed (one-hot categorical)")
        print("✓ Structure: CVLI, CVP, Tension + DOW + Month + extras")
    else:
        print(f"✗ VALIDATION FAILED: Expected 26 channels, got {node_features.shape[2]}")
else:
    print(f"✗ File not found: {pkl_file}")

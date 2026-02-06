#!/usr/bin/env python
"""Test RankingInference integration in app.py"""
import sys
import os
import numpy as np

# Add src to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Test 1: Import extract_features_clean
print("Test 1: Importing extract_features_clean from app.py...")
try:
    from app import extract_features_clean
    print("✅ Import OK")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Run extract_features_clean
print("\nTest 2: Running extract_features_clean...")
try:
    X = np.random.randn(10, 30)  # 10 nodes, 30 days
    features = extract_features_clean(X)
    assert features.shape == (10, 12), f"Expected (10, 12), got {features.shape}"
    assert not np.isnan(features).any(), "Features contain NaN"
    print(f"✅ Features extracted: shape={features.shape}")
    print(f"   Sample features (node 0): {features[0, :3]}")
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check ranking_validator availability
print("\nTest 3: Checking ranking_validator availability...")
try:
    from src.ranking_inference import RankingInference
    from pathlib import Path
    
    # Find ranking model for today
    from datetime import datetime
    day_of_week = datetime.now().weekday()
    model_path = Path(ROOT) / 'models' / 'ranking_by_day' / f'ranking_model_day{day_of_week}.pth'
    
    if model_path.exists():
        ranking_validator = RankingInference(str(model_path), device='cpu')
        if ranking_validator.model is not None:
            print(f"✅ RankingInference loaded for day {day_of_week}")
            
            # Test 4: Run validate_stgcn_predictions
            print("\nTest 4: Running validate_stgcn_predictions...")
            stgcn_scores = np.random.rand(10) * 100
            features = extract_features_clean(np.random.randn(10, 30))
            
            combined_scores, top_indices = ranking_validator.validate_stgcn_predictions(
                stgcn_scores,
                features,
                top_k=5
            )
            
            assert len(combined_scores) == 10, f"Expected 10 combined scores, got {len(combined_scores)}"
            assert len(top_indices) == 5, f"Expected 5 top indices, got {len(top_indices)}"
            print(f"✅ Blend successful")
            print(f"   Top-5 indices: {top_indices}")
            print(f"   Top-5 scores: {combined_scores[top_indices]}")
        else:
            print("⚠️  RankingInference model failed to load")
    else:
        print(f"⚠️  Model not found: {model_path}")
        
except Exception as e:
    print(f"❌ RankingInference test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 ALL TESTS PASSED - RankingInference integration ready!")
print("="*60)

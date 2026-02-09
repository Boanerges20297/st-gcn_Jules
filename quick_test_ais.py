import json
import os

BASE_DIR = os.path.dirname(__file__)

# Test 1: Check if AIS - CAPITAL.geojson exists and loads
ais_capital_path = os.path.join(BASE_DIR, 'data', 'static', 'AIS - CAPITAL.geojson')
print(f"[TEST 1] AIS - CAPITAL.geojson exists: {os.path.exists(ais_capital_path)}")

if os.path.exists(ais_capital_path):
    try:
        with open(ais_capital_path, 'r', encoding='utf-8') as f:
            ais_data = json.load(f)
            features = ais_data.get('features', [])
            print(f"[TEST 1] Loaded {len(features)} polygon features")
            if len(features) > 0:
                first = features[0]
                print(f"[TEST 1] First feature type: {first.get('geometry', {}).get('type')}")
                print(f"[TEST 1] First feature name: {first.get('properties', {}).get('name', 'N/A')}")
    except Exception as e:
        print(f"[TEST 1] ERROR: {e}")
else:
    print(f"[TEST 1] File not found: {ais_capital_path}")

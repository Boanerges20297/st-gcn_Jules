"""
Test API points with fresh Python process
"""
import subprocess
import sys
import os

# Kill any background processes
os.system("taskkill /f /im python.exe 2>/dev/null || true")

# Run test in fresh process
result = subprocess.run([
    sys.executable, '-c', '''
import sys
import os
sys.path.insert(0, os.getcwd())

from app import app
import json

app.config['TESTING'] = True

with app.test_client() as client:
    response = client.get('/api/polygons')
    data = response.get_json()
    
    features = data.get('features', [])
    print(f"✓ Total features: {len(features)}")
    
    points = sum(1 for f in features if f['geometry']['type'] == 'Point')
    polygons = sum(1 for f in features if f['geometry']['type'] == 'Polygon')
    
    print(f"  - Points: {points}")
    print(f"  - Polygons: {polygons}")
    
    if points > 0:
        print("\\n✅ MICRO-NÕES APPEARING NOW!")
    else:
        print("\\n❌ Still no micro-nodes")
'''
], capture_output=True, text=True, timeout=30)

print(result.stdout)
if result.stderr:
    # Filter to show only our custom output, not deprecation warnings
    for line in result.stderr.split('\\n'):
        if '[DEBUG]' in line or 'DETERMINISM' in line or 'MICRO' in line:
            print(line)

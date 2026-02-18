import json
import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import importlib
app_mod = importlib.import_module('app')
app = app_mod.app
normalize_name = app_mod.normalize_name
import pandas as pd

# Prepare minimal nodes_gdf and orchestrator
nodes = pd.DataFrame([
    {'name': 'Aldeota', 'region_type': 'fortaleza'}
])
nodes.index = [0]

class DummyOrch:
    def __init__(self):
        self.dates = ['2026-01-01']
    def get_combined_risk(self, exo=None):
        # return mapping normalized name -> score percent
        return {normalize_name('Aldeota'): 60.6}

app_mod.nodes_gdf = nodes
app_mod.orchestrator = DummyOrch()

with app.test_client() as c:
    resp = c.get('/api/explain/0')
    print('STATUS', resp.status_code)
    try:
        print(json.dumps(resp.get_json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print('No JSON:', resp.data)

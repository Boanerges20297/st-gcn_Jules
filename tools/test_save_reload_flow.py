import sys, os, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
import importlib
app_mod = importlib.import_module('app')
app = app_mod.app

# Dummy nodes_gdf
import pandas as pd
nodes = pd.DataFrame([{'name':'Aldeota','region_type':'fortaleza'}])
nodes.index=[0]

class DummyOrch:
    def __init__(self):
        pass
    def get_combined_risk(self, exogenous_shocks=None):
        # baseline 30 for Aldeota, if exogenous_shocks contains Aldeota increase to 90
        res = {'ALDEOTA': 30.0}
        if exogenous_shocks:
            for k,v in exogenous_shocks.items():
                if 'ALDEOTA' in k.upper():
                    res['ALDEOTA'] = 90.0
        return res

app_mod.nodes_gdf = nodes
app_mod.orchestrator = DummyOrch()

with app.test_client() as c:
    # Get baseline
    r0 = c.get('/api/risk')
    print('before save /api/risk status', r0.status_code)
    j0 = r0.get_json()
    print('before meta counts:', j0.get('meta',{})['counts'])

    # Prepare point for Aldeota
    point = {'bairro':'Aldeota','municipio':'Fortaleza','date': '2026-02-17','intensity': 1.0,'type':'confronto','lat':-3.72,'lng':-38.53}
    resp = c.post('/api/exogenous/save', json={'points':[point], 'original_text': 'test'})
    print('/api/exogenous/save', resp.status_code, resp.get_json())

    # Now call /api/risk again
    r1 = c.get('/api/risk')
    print('after save /api/risk status', r1.status_code)
    j1 = r1.get_json()
    # find Aldeota in data
    data = j1.get('data', [])
    found = [d for d in data if d.get('clean_name')=='ALDEOTA' or d.get('name','').upper().find('ALDEOTA')!=-1]
    print('found entries for Aldeota:', found)
    # inspect exogenous_events file
    exo_path = os.path.join(BASE, 'data', 'exogenous_events.json')
    print('exo file exists?', os.path.exists(exo_path))
    if os.path.exists(exo_path):
        with open(exo_path,'r',encoding='utf-8') as f:
            evs = json.load(f)
        print('last exo saved count', len(evs))

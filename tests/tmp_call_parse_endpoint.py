import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, load_data_and_models

load_data_and_models()
client = app.test_client()

payload = {'text': '01 - M20260104460 - HOMICIDIO A BALA - VITIMA DO SEXO MASCULINO - EM RESIDENCIA - RUA LUIZ MENDES XAVIER N 210 - PACAJUS (AIS25) - 01:10'}
resp = client.post('/api/exogenous/parse', json=payload)
print('STATUS:', resp.status_code)
try:
    import json
    print(json.dumps(resp.get_json(), ensure_ascii=False, indent=2))
except Exception:
    print(resp.get_data(as_text=True))

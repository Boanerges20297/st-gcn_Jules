import json
import sys
import os
# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = app.test_client()

payload = {'text': 'Relato de confronto em Fortaleza, bairro Aldeota. Houveram 2 mortos e 3 presos.'}
resp = client.post('/api/exogenous/parse', json=payload)
print('STATUS:', resp.status_code)
try:
    print('JSON:', json.dumps(resp.get_json(), ensure_ascii=False, indent=2))
except Exception as e:
    print('RESPONSE TEXT:', resp.get_data(as_text=True))

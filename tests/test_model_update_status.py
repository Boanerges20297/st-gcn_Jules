import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = app.test_client()
resp = client.get('/api/model-update-status')
print('STATUS', resp.status_code)
print(resp.get_json())

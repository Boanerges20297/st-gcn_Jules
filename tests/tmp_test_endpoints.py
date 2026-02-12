import sys
sys.path.insert(0, '.')
from app import load_data_and_models, app
import json

if __name__ == '__main__':
    load_data_and_models()
    client = app.test_client()

    print('Testing /api/explain/0')
    resp = client.get('/api/explain/0')
    print('Status:', resp.status_code)
    try:
        print(json.dumps(resp.get_json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print('No JSON:', e)

    print('\nTesting /api/simulate')
    payload = {'node_id': 0, 'severity': 0.7}
    resp2 = client.post('/api/simulate', json=payload)
    print('Status:', resp2.status_code)
    try:
        print(json.dumps(resp2.get_json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print('No JSON:', e)

import json, time
from app import app
with app.test_client() as c:
    time.sleep(8)
    for _ in range(5):
        r = c.get('/api/risk')
        if r.status_code == 200:
            break
        print("Waiting...", r.status_code)
        time.sleep(3)
    print('Status:', r.status_code)
    try:
        d = json.loads(r.data.decode('utf-8'))
        print('JSON ok, keys:', list(d.keys()))
        if 'error' in d: print('Error message:', d['error'])
    except Exception as e:
        print('JSON decode error!', e)
        print(r.data.decode('utf-8')[:200])

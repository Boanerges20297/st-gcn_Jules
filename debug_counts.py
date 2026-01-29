import json
import sys
# Ensure project root on path
sys.path.insert(0, '.')
import app
from flask import json as fj

# Call calculate_risk (returns a Flask response) inside app context
with app.app.app_context():
    resp = app.calculate_risk()
# If resp is a tuple (response, status), handle that
if isinstance(resp, tuple):
    flask_resp, status = resp
else:
    flask_resp = resp

try:
    data = flask_resp.get_json()
except Exception:
    # If get_json not available, try data attribute
    txt = flask_resp.get_data(as_text=True) if hasattr(flask_resp, 'get_data') else str(flask_resp)
    data = json.loads(txt)

items = data.get('data', [])

from collections import Counter
labels = Counter()
priority = 0
for it in items:
    lab = it.get('status_label') or 'Unknown'
    labels[lab] += 1
    if it.get('priority_cvli'):
        priority += 1

print('Total areas:', len(items))
print('By status:')
for k,v in labels.most_common():
    print(f'  {k}: {v}')
print('Priority (top percentile) count:', priority)

# Show top 10 highest risks
sorted_items = sorted(items, key=lambda x: x.get('risk_score', 0), reverse=True)
print('\nTop 10 by risk_score:')
for i, it in enumerate(sorted_items[:10]):
    print(i+1, it.get('node_id'), it.get('risk_text'), it.get('cvli_prediction_text'))

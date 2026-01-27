import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.llm_service import process_exogenous_text
import json


def test_exogenous_parsing_cases():
    p = os.path.join(os.path.dirname(__file__), '..', 'data', 'exogenous_events.json')
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cases = {
        # id -> expected fields
        '2': {'bairro': 'Granja Lisboa', 'municipio': 'FORTALEZA'},
        '3': {'municipio': 'Caucaia'},
        '4': {'bairro': 'Passaré', 'municipio': 'FORTALEZA'},
        '7': {'municipio': 'Maracanaú'}
    }

    for cid, expected in cases.items():
        item = next((it for it in data if it.get('id') == cid), None)
        assert item is not None, f"Event id {cid} not found in exogenous_events.json"
        res = process_exogenous_text(item.get('original_text', ''))
        assert isinstance(res, list) and len(res) >= 1
        evt = res[0]
        for k, v in expected.items():
            val = evt.get(k) or ''
            assert val.strip().upper() == v.strip().upper(), f"id={cid} expected {k}={v}, got {val}"

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.llm_service import process_exogenous_text
import json, traceback
p='data/exogenous_events.json'
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)
    txt='\n'.join([it.get('original_text','') for it in data])
try:
    res=process_exogenous_text(txt)
    print('TYPE:', type(res))
    print('REPR:', repr(res)[:1000])
except Exception:
    traceback.print_exc()

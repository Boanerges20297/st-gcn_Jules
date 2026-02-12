import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.llm_service import process_exogenous_text

text = "01 - M20260104460 - HOMICIDIO A BALA - VITIMA DO SEXO MASCULINO - EM RESIDENCIA - RUA LUIZ MENDES XAVIER N 210 - PACAJUS (AIS25) - 01:10"
try:
    res = process_exogenous_text(text)
    import json
    print(json.dumps(res, ensure_ascii=False, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('ERROR:', e)

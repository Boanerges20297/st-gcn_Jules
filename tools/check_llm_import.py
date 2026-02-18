import traceback
import sys
import os

# Ensure project root is on sys.path so `import src...` works when run from tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import src.llm_service as m
    print('OK: imported src.llm_service')
    print('has process_exogenous_text =', hasattr(m, 'process_exogenous_text'))
except Exception:
    traceback.print_exc()
    sys.exit(2)

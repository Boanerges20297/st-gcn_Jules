import os
import sys
sys.path.append(os.getcwd())
from src.core.orchestrator import StateOrchestrator

try:
    print("🧪 Testando carregamento do Orquestrador...")
    orch = StateOrchestrator(os.getcwd())
    print(f"✅ Especialistas carregados: {list(orch.specialists.keys())}")
    if 'interior' in orch.specialists:
        print("✅ Interior carregado com sucesso sem erros de Dtype.")
    else:
        print("❌ Interior não foi carregado. Verifique os arquivos pkl/pth.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Falha crítica no Orquestrador: {e}")

import sys
import os
import json

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.multi_agent_system import GeneralManagerAgent

print("=== INICIANDO EXECUÇÃO SINCRONA DO SISTEMA MULTI-AGENTE ===")
raw_stgcn_data = {
    "confidence_scores": [0.82, 0.76, 0.89],
    "timestamp": "2026-05-27T14:00:00"
}
user_profile = {
    "region": "Fortaleza",
    "focus": "CVLI",
    "historical_alerts": 3
}

try:
    manager = GeneralManagerAgent()
    print("Orquestrador instanciado com sucesso. Iniciando calibracao...")
    result = manager.process_and_calibrate(raw_stgcn_data, user_profile)
    print("\n[OK] CALIBRACAO CONCLUIDA COM SUCESSO!")
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as e:
    import traceback
    print("\n[ERROR] FALHA NA CALIBRACAO:")
    traceback.print_exc()

import os
import sys

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from src.core.orchestrator import StateOrchestrator

print("--- TESTING ORCHESTRATOR INITIALIZATION ---")
try:
    orch = StateOrchestrator(ROOT_DIR)
    print("\nORCHESTRATOR INITIALIZED SUCCESSFULLY")
    print(f"Loaded Specialists: {list(orch.specialists.keys())}")
except Exception as e:
    print("\nFAILED TO INITIALIZE ORCHESTRATOR")
    import traceback
    traceback.print_exc()

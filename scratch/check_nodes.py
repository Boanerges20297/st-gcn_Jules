import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.append('.')

from src.core.orchestrator import StateOrchestrator

try:
    orch = StateOrchestrator('.')
    print(f"Especialistas carregados: {list(orch.specialists.keys())}")
    
    if 'rmf' in orch.specialists:
        rmf_nodes = orch.specialists['rmf']['data']['nodes_gdf']
        matches = [n for n in rmf_nodes['name'] if 'MIGUEL' in n.upper()]
        print(f"RMF Matches: {matches}")
    else:
        print("RMF não está nos especialistas carregados!")
    
    fort_nodes = orch.specialists['fortaleza']['data']['nodes_gdf']
    matches_f = [n for n in fort_nodes['name'] if 'MIGUEL' in n.upper()]
    print(f"Fortaleza Matches: {matches_f}")

except Exception as e:
    print(f"Error: {e}")

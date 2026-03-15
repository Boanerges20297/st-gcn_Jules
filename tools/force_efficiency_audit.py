import os
import pandas as pd
from src.core.orchestrator import StateOrchestrator
from src.core.efficiency_monitor import EfficiencyMonitor

def force_evaluation():
    print("🚀 Iniciando Auditoria de Validação - 33 Canais & Nova Métrica")
    root = os.getcwd()
    
    # Load all regional metadata to simulate app.py behavior
    dfs = []
    for reg in ['fortaleza', 'rmf', 'interior']:
        path = os.path.join(root, "data", "processed", f"processed_{reg}.pkl")
        if os.path.exists(path):
            data = pd.read_pickle(path)
            if "nodes_gdf" in data:
                dfs.append(data["nodes_gdf"])
    
    nodes_gdf = pd.concat(dfs, ignore_index=True) if dfs else None
    
    orch = StateOrchestrator(root)
    monitor = EfficiencyMonitor(root, orch, nodes_gdf)
    
    metrics = monitor.run_evaluation()
    
    if metrics:
        print("\n" + "="*50)
        print("📊 RESULTADOS DA AUDITORIA")
        print("="*50)
        for reg in ['global', 'fortaleza', 'rmf', 'interior']:
            if reg in metrics:
                m = metrics[reg]
                if isinstance(m, dict) and 'p10' in m:
                    print(f"📍 {reg.upper()}:")
                    print(f"   Precisão@10: {m['p10']*100:.1f}% | GT Count: {m['active_locations']}")
                    print(f"   Top 10 Ranking: {', '.join(m.get('ranking_top10', ['N/A']))}")
                    print(f"   Ground Truth (Amostra): {', '.join(list(m.get('gt_sample', []))[:10])}")
                    print(f"   Hits no Top 10: {', '.join(m.get('hits10', []))}")
                else:
                    print(f"📍 {reg.upper()}: {m.get('status', 'N/A')}")
        print("="*50)

if __name__ == "__main__":
    force_evaluation()

import os
import sys
import pickle
import json
from datetime import datetime

# Adicionar o diretório atual ao sys.path
sys.path.append(os.getcwd())

from src.core.orchestrator import StateOrchestrator
from src.core.efficiency_monitor import EfficiencyMonitor

def main():
    print("🧪 [Teste Monitor] Inicializando componentes para teste isolado...")
    
    project_root = os.getcwd()
    
    # 1. Carregar Orquestrador
    orchestrator = StateOrchestrator(project_root)
    
    # 2. Carregar Metadados Globais para o nodes_gdf
    path = os.path.join(project_root, "data", "processed", "processed_graph_data_global.pkl")
    with open(path, "rb") as f:
        nodes_gdf = pickle.load(f).get("nodes_gdf")
    
    print(f"✅ Componentes prontos. Analisando {len(nodes_gdf)} localidades.")
    
    # 3. Instanciar e Rodar Monitor
    monitor = EfficiencyMonitor(project_root, orchestrator, nodes_gdf)
    
    print("\n🚀 [Teste Monitor] Executando avaliação regionalizada...")
    metrics = monitor.run_evaluation()
    
    if metrics:
        print("\n📊 RESULTADOS DA EFICIÊNCIA (JSON):")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        
        # Validação de campos
        if 'fortaleza' in metrics and 'p10' in metrics['fortaleza']:
            print(f"\n🎯 Sucesso: Eficiência de Fortaleza capturada ({metrics['fortaleza']['p10']*100:.1f}%)")
        if 'rmf' in metrics and 'p10' in metrics['rmf']:
            print(f"🎯 Sucesso: Eficiência da RMF capturada ({metrics['rmf']['p10']*100:.1f}%)")
    else:
        print("\n❌ Falha: O monitor não retornou métricas. Verifique os eventos exógenos.")

if __name__ == "__main__":
    main()

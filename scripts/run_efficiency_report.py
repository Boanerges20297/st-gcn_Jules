import os
import sys
import json

# Adicionar raiz ao path
sys.path.append(os.getcwd())

from src.core.orchestrator import StateOrchestrator
from src.core.efficiency_monitor import EfficiencyMonitor

def main():
    print("📊 Iniciando Diagnóstico de Eficiência do Report Preview...")
    
    try:
        # 1. Inicializar Orquestrador (Carrega os modelos atuais)
        print("   - Carregando modelos especialistas...")
        orchestrator = StateOrchestrator(os.getcwd())
        
        # 2. Inicializar Monitor
        monitor = EfficiencyMonitor(os.getcwd(), orchestrator, None)
        
        # 3. Executar Avaliação
        print("   - Comparando previsões com eventos reais dos últimos 7 dias...")
        metrics = monitor.run_evaluation()
        
        if not metrics:
            print("❌ Não foi possível gerar o relatório. Verifique se há eventos recentes em 'data/exogenous_events.json'.")
            return

        # 4. Exibir Relatório
        print("\n" + "="*60)
        print(f"📄 RELATÓRIO DE EFICÁCIA - DATA: {metrics.get('date')}")
        print("="*60)
        
        # Global
        glob = metrics.get('global', {})
        print(f"\n🌍 ESTADO DO CEARÁ (Global - 299 Localidades)")
        print(f"   - Eventos Reais Capturados: {glob.get('total_events', 0)}")
        print(f"   - Localidades Ativas: {glob.get('active_locations', 0)}")
        print(f"   - Precisão Top 5 (P5):  {glob.get('p5', 0)*100:.1f}%")
        print(f"   - Precisão Top 10 (P10): {glob.get('p10', 0)*100:.1f}%")
        print(f"   - Precisão Top 20 (P20): {glob.get('p20', 0)*100:.1f}%")
        
        # Regionais
        for reg in ['fortaleza', 'rmf', 'interior']:
            r_data = metrics.get(reg, {})
            if not r_data: continue
            print(f"\n📍 {reg.upper()}")
            print(f"   - P5: {r_data.get('p5', 0)*100:.1f}% | P10: {r_data.get('p10', 0)*100:.1f}%")
            if r_data.get('hits10'):
                print(f"   - Acertos (Top 10): {', '.join(r_data['hits10'][:5])}...")

        print("\n" + "="*60)
        print("✅ Relatório gerado e salvo em logs/efficiency_history.json")

    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
ANÁLISE DE SCRIPTS - Identificar úteis vs históricos
"""

import os
from pathlib import Path

BASE_DIR = Path('scripts')

scripts_info = {
    'CRÍTICOS (Manutenção diária)': {
        'cleanup_outputs.py': 'Limpeza periódica de arquivos temporários',
        'final_summary.py': 'Resumo final do status do sistema',
        'monitor_training.py': 'Monitorar progresso de treinamento',
        'session_summary.py': 'Resumo da sessão de desenvolvimento',
    },
    
    'ANÁLISE E VALIDAÇÃO (Diagnóstico)': {
        'analyze_outputs.py': 'Analisar espaço e organização de outputs/',
        'validate_tcp_territory.py': 'Validar território do TCP',
        'compare_cv_tcp.py': 'Comparar CV vs TCP territorialmente',
        'identify_conflicts.py': 'Identificar conflitos de atribuição',
    },
    
    'HISTÓRICOS (Fase de integração - podem deletar)': {
        'apply_cv_territory_assignments.py': 'Aplicar CV (versão inicial)',
        'apply_all_factions.py': 'Aplicar facções (versão 1)',
        'apply_all_factions_fixed.py': 'Aplicar facções (versão 2)',
        'apply_manual_faction_assignments.py': 'Atribuição manual antiga',
        'extract_territory_from_kml.py': 'Extração KML (versão 1)',
        'extract_all_factions_territory.py': 'Extração facções (versão 1)',
        'extract_factions_from_kml.py': 'Extração KML (versão 2)',
        'extract_factions_from_kml_advanced.py': 'Extração KML (versão 3)',
        'extract_node_bairro_faction.py': 'Extração antiga de bairros',
        'faction_color_geo_mapping.py': 'Mapeamento de cores (obsoleto)',
        'faction_reassignment_apply.py': 'Reatribuição antiga',
        'sync_faction_to_app.py': 'Sincronização antiga',
        'update_model_data_factions.py': 'Atualização antiga de modelo',
        'force_disputa_assignment.py': 'Força atribuição DISPUTA (uma vez)',
        'retrain_with_corrected_data.py': 'Wrapper de treinamento (redundante)',
        'analyze_faction_territories.py': 'Análise antiga de territórios',
        'analyze_faction_territories_corrected.py': 'Análise antiga (v2)',
        'assign_faction_from_intel.py': 'Atribuição por inteligência',
        'integrate_kml_factions_to_graph.py': 'Integração antiga',
        'integrate_reassignments.py': 'Integração de reatribuições',
        'map_ais_to_factions.py': 'Mapeamento AIS antigo',
        'match_intel_to_nodes.py': 'Correspondência com inteligência',
    },
    
    'INSPEÇÃO/DEBUG (Utilities)': {
        'inspect_processed_pickle.py': 'Inspecionar dados do pickle',
    },
    
    'DADOS (JSON/Resultados antigos)': {
        'prison_by_bairro_results.json': 'Resultados de prisão (histórico)',
        'prison_correlation.json': 'Correlação de prisões (histórico)',
        'prison_vs_predictions_results.json': 'Validação de prisões (histórico)',
    },
    
    'SUBDIRETÓRIOS': {
        'debug/': 'Pasta de debug',
        'tests/': 'Pasta de testes',
        'training/': 'Scripts de treinamento',
        'tuning/': 'Scripts de tuning',
        'utilities/': 'Utilitários diversos',
        '__pycache__/': 'Cache Python (seguro deletar)',
    },
    
    'SCRIPTS SHELL': {
        'auto_merge.ps1': 'Auto-merge em PowerShell (histórico)',
    },
}

print("\n" + "="*100)
print("📊 ANÁLISE DE SCRIPTS")
print("="*100)

total_size = 0
total_files = 0

for category, scripts in scripts_info.items():
    print(f"\n{'='*100}")
    print(f"📁 {category}")
    print(f"{'='*100}")
    
    category_size = 0
    for filename, description in scripts.items():
        filepath = BASE_DIR / filename
        
        if filepath.is_file() and filepath.exists():
            size = filepath.stat().st_size / 1024
            category_size += size
            total_size += size
            total_files += 1
            
            status = "✅" if "CRÍTICO" in category or "ANÁLISE" in category else "⚠️ "
            print(f"  {status} {filename:45s} ({size:8.1f} KB) - {description}")
        elif filepath.is_dir():
            print(f"  📂 {filename:45s} (diretório) - {description}")
        else:
            print(f"  ❌ {filename:45s} (NÃO ENCONTRADO)")
    
    if category_size > 0:
        print(f"\n  📊 Total da categoria: {category_size:.1f} KB")

print(f"\n{'='*100}")
print(f"📊 RESUMO GERAL")
print(f"{'='*100}")
print(f"Total de arquivos: {total_files}")
print(f"Tamanho total: {total_size:.1f} KB ({total_size/1024:.2f} MB)")

print(f"\n{'='*100}")
print(f"✅ SCRIPTS CRÍTICOS (NÃO DELETAR):")
print(f"{'='*100}")
critical = [
    'cleanup_outputs.py ← Limpeza periódica',
    'final_summary.py ← Status do sistema',
    'analyze_outputs.py ← Diagnóstico',
    'validate_tcp_territory.py ← Validação',
    'identify_conflicts.py ← Detecção de conflitos',
]
for item in critical:
    print(f"   🔴 {item}")

print(f"\n{'='*100}")
print(f"⚠️  ARQUIVOS CANDIDATOS PARA DELEÇÃO:")
print(f"{'='*100}")
print("""
Versões antigas de extração/aplicação (20 scripts):
   - apply_cv_territory_assignments.py (v1)
   - apply_all_factions.py (v1)
   - apply_all_factions_fixed.py (v2)
   - extract_territory_from_kml.py (v1)
   - extract_factions_from_kml.py (v2)
   - extract_factions_from_kml_advanced.py (v3)
   - [+ 14 scripts históricos]

Dados antigos (3 JSON):
   - prison_by_bairro_results.json
   - prison_correlation.json
   - prison_vs_predictions_results.json

Subdiretórios (5):
   - debug/
   - tests/
   - training/
   - tuning/
   - utilities/
   - __pycache__/

Total estimado: 0.98 MB a liberar
""")

print(f"\n{'='*100}\n")

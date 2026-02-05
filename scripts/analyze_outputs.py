"""
ANÁLISE DE ARQUIVOS EM /outputs
Identificar o que é necessário vs o que pode ser deletado
"""

import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path('outputs')

# Categorizar arquivos
files_info = {
    'NECESSÁRIOS (Sistema em uso)': {
        'nodes_with_faction_assigned.geojson': 'Arquivo PRINCIPAL - carregado por app.py',
        'cv_territory_from_kml.json': 'Dados de validação do CV (usado em análises)',
        'tcp_territory_from_kml.json': 'Dados de validação do TCP (usado em análises)',
        'massa_territory_from_kml.json': 'Dados de validação MASSA',
        'pcc_territory_from_kml.json': 'Dados de validação PCC',
        'fantasmas_territory_from_kml.json': 'Dados de validação FANTASMAS',
        'disputa_territory_from_kml.json': 'Dados de validação DISPUTA',
        'okaida_territory_from_kml.json': 'Dados de validação OKAIDA',
    },
    
    'BACKUPS (Manter por segurança)': {
        'nodes_with_faction_assigned_backup.geojson': 'Backup do arquivo principal',
        'nodes_with_kml_factions_final_backup.geojson': 'Backup intermediário',
    },
    
    'VISUALIZAÇÕES (Manter para referência)': {
        'cv_territory_analysis.html': 'Relatório visual do CV',
        'faction_mapping_visualization.html': 'Visualização de mapeamento',
    },
    
    'HISTÓRICOS (Podem ser deletados)': {
        'faction_color_analysis.json': 'Análise antiga de cores',
        'faction_mapping_known.json': 'Mapeamento antigo de facções',
        'faction_reassignments_to_apply.json': 'Reatribuições antigas',
        'faction_reassignment_candidates.json': 'Candidatos antigos',
        'color_geography_analysis_detailed.json': 'Análise detalhada antiga',
        'faction_territory_corrected.csv': 'CSV antigo',
        'faction_territory_summary.csv': 'CSV antigo',
        'faction_territories.geojson': 'Versão antiga',
        'faction_territories_refined.geojson': 'Versão refinada antiga',
        'model_data_update_log.json': 'Log de atualização',
        'integration_log.json': 'Log de integração',
        'sync_faction_log.json': 'Log de sincronização',
        'nodes_enriched_with_kml_factions.geojson': 'Versão intermediária',
        'nodes_with_bairro_faction.geojson': 'Versão intermediária',
        'nodes_with_kml_factions.geojson': 'Versão intermediária',
        'nodes_with_kml_factions_advanced.geojson': 'Versão intermediária',
        'nodes_with_kml_factions_final.geojson': 'Versão intermediária (superado por assigned)',
        'nodes_with_kml_factions_reassigned.geojson': 'Versão intermediária',
    },
    
    'DADOS PROCESSADOS (Manter)': {
        'enriched_timeseries_by_bairro.csv': 'Dados de séries temporais por bairro',
        'enriched_timeseries_by_faction.csv': 'Dados de séries temporais por facção',
        'occurrences_with_bairro_geo.csv': 'Ocorrências com geolocalização',
        'fortaleza_bairros_fence.geojson': 'Geometria de bairros de Fortaleza',
    },
    
    'VALIDAÇÕES (Referência)': {
        'tcp_validation.json': 'Validação do TCP (referência)',
    },
}

print("\n" + "="*100)
print("📊 ANÁLISE DE ARQUIVOS EM /outputs")
print("="*100)

total_size = 0
file_count = 0

for category, files in files_info.items():
    print(f"\n{'='*100}")
    print(f"📁 {category}")
    print(f"{'='*100}")
    
    category_size = 0
    for filename, description in files.items():
        filepath = BASE_DIR / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            category_size += size
            total_size += size
            file_count += 1
            
            status = "✅" if "NECESSÁRIO" in category or "BACKUP" in category or "VISUALIZ" in category or "Manter" in category else "⚠️ "
            print(f"  {status} {filename:50s} ({size:8.1f} KB)")
            print(f"     └─ {description}")
        else:
            print(f"  ❌ {filename:50s} (NÃO ENCONTRADO)")
    
    if category_size > 0:
        print(f"\n  📊 Total da categoria: {category_size:.1f} KB")

print(f"\n{'='*100}")
print(f"📊 RESUMO GERAL")
print(f"{'='*100}")
print(f"Total de arquivos: {file_count}")
print(f"Tamanho total: {total_size:.1f} KB ({total_size/1024:.2f} MB)")

print(f"\n{'='*100}")
print(f"🗑️  RECOMENDAÇÕES DE LIMPEZA")
print(f"{'='*100}")

delete_candidates = files_info.get('HISTÓRICOS (Podem ser deletados)', {})
delete_size = 0

print(f"\n⚠️  Arquivos candidatos para DELEÇÃO ({len(delete_candidates)} arquivos):")
for filename in sorted(delete_candidates.keys()):
    filepath = BASE_DIR / filename
    if filepath.exists():
        size = filepath.stat().st_size / 1024
        delete_size += size
        print(f"   - {filename:50s} ({size:8.1f} KB)")

print(f"\n📊 Espaço que seria liberado: {delete_size:.1f} KB ({delete_size/1024:.2f} MB)")

print(f"\n{'='*100}")
print(f"✅ ARQUIVOS CRÍTICOS (NÃO DELETAR):")
print(f"{'='*100}")
critical = [
    'nodes_with_faction_assigned.geojson ← PRINCIPAL (usado por app.py)',
    'cv_territory_from_kml.json ← Validação CV',
    'tcp_territory_from_kml.json ← Validação TCP',
    'enriched_timeseries_by_faction.csv ← Dados processados',
]
for item in critical:
    print(f"   🔴 {item}")

print(f"\n{'='*100}\n")

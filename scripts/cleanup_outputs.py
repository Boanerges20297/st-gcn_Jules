"""
Script de Limpeza de /outputs
Remove APENAS arquivos intermediários e históricos que não estão sendo usados
"""

import os
from pathlib import Path

BASE_DIR = Path('outputs')

# Lista CONSERVADORA de arquivos para deletar
# Apenas versões intermediárias e análises antigas
DELETE_LIST = [
    # Análises e cores antigas (foram usadas para gerar as atribuições correntes)
    'color_geography_analysis_detailed.json',
    'faction_color_analysis.json',
    'faction_mapping_known.json',
    'faction_reassignment_candidates.json',
    'faction_reassignments_to_apply.json',
    
    # Mapeamentos de território antigos (substituídos pelos _from_kml.json)
    'faction_territories.geojson',
    'faction_territories_refined.geojson',
    
    # Versões intermediárias antigas de nodes (substituídas por nodes_with_faction_assigned.geojson)
    'nodes_enriched_with_kml_factions.geojson',
    'nodes_with_bairro_faction.geojson',
    'nodes_with_kml_factions.geojson',
    'nodes_with_kml_factions_advanced.geojson',
    'nodes_with_kml_factions_final.geojson',  # Superado por nodes_with_faction_assigned.geojson
    'nodes_with_kml_factions_reassigned.geojson',
    
    # CSVs antigos de território (não estão sendo usados)
    'faction_territory_corrected.csv',
    'faction_territory_summary.csv',
    
    # Logs antigos (informativos, não necessários para produção)
    'model_data_update_log.json',
    'integration_log.json',
    'sync_faction_log.json',
]

print("\n" + "="*80)
print("🗑️  LIMPEZA DE /outputs - MODO CONSERVADOR")
print("="*80)

print("\n📋 Arquivos a serem deletados:")
total_size = 0
deleted_count = 0

for filename in sorted(DELETE_LIST):
    filepath = BASE_DIR / filename
    if filepath.exists():
        size = filepath.stat().st_size / 1024
        total_size += size
        print(f"   ❌ {filename:50s} ({size:8.1f} KB)")
        
        # Deletar
        try:
            filepath.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"      ⚠️  Erro ao deletar: {e}")
    else:
        print(f"   ⭕ {filename:50s} (já não existe)")

print(f"\n📊 RESULTADO:")
print(f"   ✅ Arquivos deletados: {deleted_count}/{len(DELETE_LIST)}")
print(f"   💾 Espaço liberado: {total_size:.1f} KB ({total_size/1024:.2f} MB)")

print("\n✅ ARQUIVOS PRESERVADOS (CRÍTICOS):")
critical = [
    'nodes_with_faction_assigned.geojson',
    'nodes_with_faction_assigned_backup.geojson',
    'nodes_with_kml_factions_final_backup.geojson',
    'cv_territory_from_kml.json',
    'tcp_territory_from_kml.json',
    'massa_territory_from_kml.json',
    'pcc_territory_from_kml.json',
    'fantasmas_territory_from_kml.json',
    'disputa_territory_from_kml.json',
    'okaida_territory_from_kml.json',
    'cv_territory_analysis.html',
    'faction_mapping_visualization.html',
    'enriched_timeseries_by_bairro.csv',
    'enriched_timeseries_by_faction.csv',
    'occurrences_with_bairro_geo.csv',
    'fortaleza_bairros_fence.geojson',
]

for filename in critical:
    filepath = BASE_DIR / filename
    if filepath.exists():
        size = filepath.stat().st_size / 1024
        print(f"   ✅ {filename:50s} ({size:8.1f} KB)")

print("\n" + "="*80)
print("✨ LIMPEZA CONCLUÍDA!")
print("="*80 + "\n")

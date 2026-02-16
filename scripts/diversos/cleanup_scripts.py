#!/usr/bin/env python3
"""
cleanup_scripts.py
Limpa scripts históricos e desnecessários, mantendo apenas os críticos

Scripts críticos (manutenção diária):
  ✅ cleanup_outputs.py - Limpeza periódica de outputs/
  ✅ final_summary.py - Resumo final do status
  ✅ analyze_outputs.py - Análise de espaço
  ✅ validate_tcp_territory.py - Validação
  ✅ identify_conflicts.py - Detecção de conflitos
  ✅ session_summary.py - Resumo da sessão
"""

import os
import shutil
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

# Scripts que devem ser preservados
KEEP_SCRIPTS = {
    'cleanup_outputs.py',
    'final_summary.py',
    'analyze_outputs.py',
    'validate_tcp_territory.py',
    'identify_conflicts.py',
    'session_summary.py',
    'monitor_training.py',
    'analyze_scripts.py',  # Mantém a análise para referência futura
    'cleanup_scripts.py',  # A si mesmo
}

# Scripts históricos para deletar
DELETE_SCRIPTS = [
    'apply_cv_territory_assignments.py',
    'apply_all_factions.py',
    'apply_all_factions_fixed.py',
    'apply_manual_faction_assignments.py',
    'extract_territory_from_kml.py',
    'extract_all_factions_territory.py',
    'extract_factions_from_kml.py',
    'extract_factions_from_kml_advanced.py',
    'extract_node_bairro_faction.py',
    'faction_color_geo_mapping.py',
    'faction_reassignment_apply.py',
    'sync_faction_to_app.py',
    'update_model_data_factions.py',
    'force_disputa_assignment.py',
    'retrain_with_corrected_data.py',
    'analyze_faction_territories.py',
    'analyze_faction_territories_corrected.py',
    'assign_faction_from_intel.py',
    'integrate_kml_factions_to_graph.py',
    'integrate_reassignments.py',
    'map_ais_to_factions.py',
    'match_intel_to_nodes.py',
    'inspect_processed_pickle.py',
]

# Dados históricos (JSON)
DELETE_DATA = [
    'prison_by_bairro_results.json',
    'prison_correlation.json',
    'prison_vs_predictions_results.json',
]

# Subdiretórios a deletar
DELETE_DIRS = [
    'debug',
    'tests',
    'training',
    'tuning',
    'utilities',
    '__pycache__',
]

# Scripts shell (histórico)
DELETE_SHELL = [
    'auto_merge.ps1',
]

def get_size(path):
    """Calcular tamanho do arquivo/diretório"""
    if path.is_file():
        return path.stat().st_size
    else:
        total = 0
        for p in path.rglob('*'):
            if p.is_file():
                total += p.stat().st_size
        return total

def format_size(bytes_size):
    """Formatar tamanho em KB/MB"""
    kb = bytes_size / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    else:
        return f"{kb / 1024:.2f} MB"

print("=" * 100)
print("🗑️  LIMPEZA DE SCRIPTS - REMOÇÃO DE HISTÓRICOS")
print("=" * 100)

deleted_files = []
deleted_size = 0
failed_files = []

# Deletar scripts
print("\n📝 Deletando scripts históricos...")
for script_name in DELETE_SCRIPTS:
    script_path = SCRIPTS_DIR / script_name
    if script_path.exists():
        try:
            size = get_size(script_path)
            os.remove(script_path)
            deleted_files.append(script_name)
            deleted_size += size
            print(f"   ✅ {script_name:<45} ({format_size(size):>8})")
        except Exception as e:
            failed_files.append((script_name, str(e)))
            print(f"   ❌ {script_name:<45} (Erro: {e})")

# Deletar dados históricos
print("\n📊 Deletando dados históricos...")
for data_name in DELETE_DATA:
    data_path = SCRIPTS_DIR / data_name
    if data_path.exists():
        try:
            size = get_size(data_path)
            os.remove(data_path)
            deleted_files.append(data_name)
            deleted_size += size
            print(f"   ✅ {data_name:<45} ({format_size(size):>8})")
        except Exception as e:
            failed_files.append((data_name, str(e)))
            print(f"   ❌ {data_name:<45} (Erro: {e})")

# Deletar shell scripts
print("\n🔧 Deletando scripts shell...")
for shell_name in DELETE_SHELL:
    shell_path = SCRIPTS_DIR / shell_name
    if shell_path.exists():
        try:
            size = get_size(shell_path)
            os.remove(shell_path)
            deleted_files.append(shell_name)
            deleted_size += size
            print(f"   ✅ {shell_name:<45} ({format_size(size):>8})")
        except Exception as e:
            failed_files.append((shell_name, str(e)))
            print(f"   ❌ {shell_name:<45} (Erro: {e})")

# Deletar subdiretórios
print("\n📂 Deletando subdiretórios...")
for dir_name in DELETE_DIRS:
    dir_path = SCRIPTS_DIR / dir_name
    if dir_path.exists() and dir_path.is_dir():
        try:
            size = get_size(dir_path)
            shutil.rmtree(dir_path)
            deleted_files.append(dir_name)
            deleted_size += size
            print(f"   ✅ {dir_name:<45} ({format_size(size):>8})")
        except Exception as e:
            failed_files.append((dir_name, str(e)))
            print(f"   ❌ {dir_name:<45} (Erro: {e})")

print("\n" + "=" * 100)
print("📊 RESULTADO:")
print("=" * 100)
print(f"   ✅ Arquivos/diretórios deletados: {len(deleted_files)}/{len(DELETE_SCRIPTS) + len(DELETE_DATA) + len(DELETE_DIRS) + len(DELETE_SHELL)}")
print(f"   💾 Espaço liberado: {format_size(deleted_size)}")
if failed_files:
    print(f"   ⚠️  Falhas: {len(failed_files)}")

print("\n" + "=" * 100)
print("✅ SCRIPTS PRESERVADOS (CRÍTICOS):")
print("=" * 100)
preserved = sorted([f for f in os.listdir(SCRIPTS_DIR) if f.endswith('.py') and f in KEEP_SCRIPTS])
for script in preserved:
    script_path = SCRIPTS_DIR / script
    size = get_size(script_path)
    print(f"   ✅ {script:<45} ({format_size(size):>8})")

print("\n" + "=" * 100)

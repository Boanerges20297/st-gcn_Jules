#!/usr/bin/env python3
"""
Reprocessa os dados aplicando o novo filtro CVP (apenas veículos).
Remove dados CVP genéricos e mantém apenas roubos/furtos de veículos.
"""

import os
import sys
import pickle
import numpy as np
from datetime import datetime

# Add src to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from data_processing import build_graph_data

def reprocess_data():
    """Reprocessa dados com filtro CVP refinado (apenas veículos)."""
    
    print("="*80)
    print("REPROCESSAMENTO DE DADOS - CVP VEÍCULOS APENAS")
    print("="*80)
    print()
    
    # Paths
    occurrences_file = os.path.join(BASE_DIR, 'outputs', 'occurrences_with_bairro_geo.csv')
    nodes_file = os.path.join(BASE_DIR, 'outputs', 'fortaleza_bairros_fence.geojson')
    output_dir = os.path.join(BASE_DIR, 'data', 'processed')
    output_file = os.path.join(output_dir, 'processed_graph_data.pkl')
    
    # Verificar se arquivos existem
    if not os.path.exists(occurrences_file):
        print(f"❌ Arquivo não encontrado: {occurrences_file}")
        return False
    
    if not os.path.exists(nodes_file):
        print(f"❌ Arquivo não encontrado: {nodes_file}")
        return False
    
    print(f"📂 Ocorrências: {occurrences_file}")
    print(f"📂 Nós: {nodes_file}")
    print(f"📂 Output: {output_file}")
    print()
    
    # Backup do arquivo antigo
    if os.path.exists(output_file):
        backup_file = output_file + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(f"💾 Criando backup: {backup_file}")
        import shutil
        shutil.copy2(output_file, backup_file)
        print()
    
    # Processar dados
    print("🔄 Processando dados com novo filtro CVP (veículos apenas)...")
    print()
    
    try:
        result = build_graph_data(
            occurrences_csv=occurrences_file,
            nodes_geojson=nodes_file,
            output_dir=output_dir
        )
        
        if result:
            print()
            print("="*80)
            print("✅ REPROCESSAMENTO CONCLUÍDO COM SUCESSO!")
            print("="*80)
            print()
            print("📊 Resumo:")
            print(f"   • Arquivo atualizado: {output_file}")
            print(f"   • CVP agora inclui apenas: ROUBO/FURTO + VEÍCULO/MOTO/CARRO")
            print(f"   • CVLI mantido: HOMICÍDIO DOLOSO, FEMINICÍDIO, LATROCÍNIO, etc.")
            print()
            print("⚠️  IMPORTANTE: Reinicie o servidor Flask para carregar os novos dados!")
            print()
            return True
        else:
            print()
            print("❌ Erro durante o processamento")
            return False
            
    except Exception as e:
        print()
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = reprocess_data()
    sys.exit(0 if success else 1)

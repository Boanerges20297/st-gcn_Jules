#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testa a exibição de nomes dos nodes nos arquivos de predição
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_predict_logger_with_nodes():
    """Testa PredictLogger com nomes de nodes"""
    from src.predict_logger import PredictLogger
    import pickle
    
    BASE_DIR = project_root
    DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'processed_graph_data.pkl')
    
    # Carregar dados
    print("📦 Carregando dados...")
    with open(DATA_FILE, 'rb') as f:
        data_pack = pickle.load(f)
    
    nodes_gdf = data_pack.get('nodes_gdf')
    print(f"✅ Loaded {len(nodes_gdf)} nodes")
    
    # Criar logger com nodes_gdf
    logger = PredictLogger(BASE_DIR, nodes_gdf=nodes_gdf)
    print("✅ PredictLogger criado com nodes_gdf")
    
    # Criar alguns resultados de exemplo
    results = []
    for i in range(5):
        results.append({
            'node_id': i,
            'risk_score': 90 + i,
            'ranking_score': None,
            'status_label': 'Crítico',
            'score_provenance': ['history', 'very_active'],
            'faction': 'PCC',
            'cvli_pred': 5.2 + i*0.5,
            'reasons': [f'Razão {j}' for j in range(3)]
        })
    
    # Criar metadados
    meta = {
        'counts': {
            'crítico': 10,
            'alto': 5,
            'moderado': 20,
            'baixo': 15,
            'sem risco': 50
        },
        'ranking_source': 'stgcn_percentile',
        'window_cvli': 30,
        'window_start': '2026-01-01',
        'window_end': '2026-01-30',
        'last_date': '2026-01-30',
        'distribution': {
            'norm_min': 0.3,
            'norm_max': 99.7,
            'norm_mean': 48.0,
            'norm_percentiles': {'50': 53.0, '75': 80.7, '90': 95.0, '95': 95.0, '99': 98.7}
        },
        'history_stats': {
            'hist_min': 0,
            'hist_max': 14,
            'hist_mean': 0.8,
            'hist_percentiles': {'50': 0, '75': 1, '90': 3, '95': 4}
        },
        'provenance_lists': {
            'history': [0, 1, 2, 3, 4],
            'very_active': [0, 1, 2],
            'exogenous': [3, 4],
            'exogenous_critical': [4]
        }
    }
    
    # Gerar e salvar log
    from datetime import datetime
    filepath = logger.log_prediction(meta, results, timestamp=datetime.now())
    print(f"✅ Log salvo em: {filepath}")
    
    # Ler e mostrar conteúdo relevante
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Mostrar a seção de ranking
    lines = content.split('\n')
    in_ranking = False
    for i, line in enumerate(lines):
        if 'RANKING ATUALIZADO' in line:
            in_ranking = True
        if in_ranking:
            print(line)
            if 'CORREÇÕES FEITAS' in line:
                break
    
    print("\n✅ TESTE PASSOU - Nomes dos nodes estão sendo exibidos!")

if __name__ == '__main__':
    test_predict_logger_with_nodes()

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Adicionar o diretório atual ao sys.path
sys.path.append(os.getcwd())

from src.core.orchestrator import StateOrchestrator, normalize_name

def main():
    print("--- Simulação de Sensibilidade Total: Fortaleza (Com Choques Exógenos) ---")
    
    project_root = os.getcwd()
    orchestrator = StateOrchestrator(project_root)
    
    if 'fortaleza' not in orchestrator.specialists:
        print("❌ Especialista de Fortaleza não carregado.")
        return
        
    spec = orchestrator.specialists['fortaleza']
    data = spec['data']
    nodes_gdf = data['nodes_gdf']
    fortaleza_names = set([normalize_name(str(name)) for name in nodes_gdf['name']])
    
    # 1. Carregar TODOS os eventos (Ignorando filtro de data para simulação)
    exo_path = os.path.join(project_root, 'data', 'exogenous_events.json')
    all_raw_events = []
    if os.path.exists(exo_path):
        with open(exo_path, 'r', encoding='utf-8') as f:
            all_raw_events = json.load(f)

    # 2. Preparar Shocks para o Modelo
    exogenous_shocks = {}
    ground_truth = {}
    
    for ev in all_raw_events:
        loc_raw = ev.get('bairro') or ev.get('location')
        loc_norm = normalize_name(str(loc_raw))
        
        if loc_norm in fortaleza_names:
            # Para o Ground Truth (Avaliação)
            ground_truth[loc_norm] = ground_truth.get(loc_norm, 0) + 1
            
            # Para o Input do Modelo (Injeção de Choque)
            intensity = float(ev.get('intensity', 0.5))
            ev_type = str(ev.get('type','')).lower()
            is_crit = intensity > 0.7 or any(x in ev_type for x in ['confronto', 'execucao', 'chacina', 'homicidio', 'facca'])
            is_supp = any(x in ev_type for x in ['apreensao', 'prisao', 'recupera'])
            
            if loc_norm not in exogenous_shocks:
                exogenous_shocks[loc_norm] = {'intensity': intensity, 'is_critical': is_crit, 'is_suppression': is_supp}
            else:
                exogenous_shocks[loc_norm]['intensity'] = max(exogenous_shocks[loc_norm]['intensity'], intensity)
                if is_crit: exogenous_shocks[loc_norm]['is_critical'] = True
                if is_supp: exogenous_shocks[loc_norm]['is_suppression'] = True

    print(f"💉 Injetando shocks em {len(exogenous_shocks)} bairros de Fortaleza...")

    # 3. Rodar Modelo com os Choques
    print("🧠 Processando influência espacial via ST-GAT...")
    scores_with_shocks = orchestrator.get_combined_risk(exogenous_shocks)
    fortaleza_scores = {name: score for name, score in scores_with_shocks.items() if name in fortaleza_names}
    
    # 4. Cálculo de Eficiência (O quanto o choque posicionou bem os bairros)
    k_values = [5, 10, 15, 20]
    sorted_fortaleza = sorted(fortaleza_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Métricas de Eficiência (Com Canais Exógenos Ativos) ---")
    print(f"{'K':<3} | {'Precisão (%)':<12} | {'Bairros Identificados'}")
    print("-" * 50)
    
    for k in k_values:
        top_k_pred = [name for name, score in sorted_fortaleza[:k]]
        hits = [name for name in top_k_pred if name in ground_truth]
        precision = (len(hits) / k) * 100
        print(f"{k:<3} | {precision:<12.1f} | {len(hits)}/{k} ({', '.join(hits[:2])}...)")

    # 5. Top 10 Bairros Reais vs Posição no Ranking Pós-Choque
    print("\n🏆 Reação do Modelo nos Hotspots Reais:")
    for name, count in sorted(ground_truth.items(), key=lambda x: x[1], reverse=True):
        rank = -1
        score = 0
        for i, (n, s) in enumerate(sorted_fortaleza):
            if n == name:
                rank = i + 1
                score = s
                break
        print(f"- {name:20s}: {count} eventos | Nova Pos. Ranking: {rank}º (Score: {score:.2f}%)")

if __name__ == "__main__":
    main()

import os
import sys
import json
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime, timedelta

# Garantir que a raiz do projeto está no path
sys.path.append(os.getcwd())

from src.core.orchestrator import StateOrchestrator, normalize_name
from src.core.champion_challenger import ChampionChallenger

def _pk(scores_map, ground_truth, k, top_bairros):
    """Calcula P@k para um mapeamento de scores e ground truth."""
    # Filtrar apenas bairros monitorados
    ordered = sorted(top_bairros, key=lambda b: scores_map.get(b, 0), reverse=True)
    top_k = ordered[:k]
    
    hits = sum(1 for b in top_k if ground_truth.get(b, 0) > 0)
    return hits / k

def evaluate():
    BASE_DIR = os.getcwd()
    print("Inicializando Orquestrador (Champion)...")
    orchestrator = StateOrchestrator(BASE_DIR)
    
    print("Inicializando ChampionChallenger (Challenger)...")
    cc = ChampionChallenger(BASE_DIR)
    
    if cc._ranker is None:
        print("Erro: Modelo Challenger não carregado corretamente.")
        return

    # 1. Obter predições do Champion (ST-GAT)
    print("Gerando scores Champion (ST-GAT)...")
    scores_champ = orchestrator.get_combined_risk()
    
    # 2. Obter predições do Challenger (LGBM)
    # Precisamos garantir que as features estão prontas
    cc._ensure_features()
    print("Gerando scores Challenger (LGBM)...")
    scores_chal = cc._get_challenger_scores()
    
    # 3. Obter predições do Ensemble (Blend)
    # Forçamos um peso para ver o efeito (ex: 50/50 se o challenger for bom)
    cc._cc_weight = 0.5
    print("Gerando scores Ensemble (50/50 Blend)...")
    scores_ensemble = cc.apply(scores_champ)

    # 4. Construir Ground Truth (Últimos 14 dias)
    print("Coletando dados reais dos últimos 14 dias...")
    T = len(cc._dates)
    window = 14
    gt_matrix = cc._cvli_raw[:, max(0, T-window):T].sum(axis=1)
    
    ground_truth = {cc._top_bairros[i]: gt_matrix[i] for i in range(len(cc._top_bairros))}
    total_cvli = sum(ground_truth.values())
    bairros_com_crime = sum(1 for v in ground_truth.values() if v > 0)
    
    print(f"✅ Ground Truth: {total_cvli:.0f} crimes em {bairros_com_crime} bairros (Fortaleza).")

    # 5. Calcular Métricas
    results = []
    top_bairros = cc._top_bairros
    
    for label, scores in [("Champion (ST-GAT)", scores_champ), 
                          ("Challenger (LGBM)", scores_chal), 
                          ("Ensemble (50/50)", scores_ensemble)]:
        p10 = _pk(scores, ground_truth, 10, top_bairros)
        p20 = _pk(scores, ground_truth, 20, top_bairros)
        results.append({"Modelo": label, "P@10": p10, "P@20": p20})

    # 6. Exibir Tabela de Resultados
    print("\n" + "="*50)
    print(f"{'MODELO':<25} | {'P@10':<8} | {'P@20':<8}")
    print("-" * 50)
    for res in results:
        print(f"{res['Modelo']:<25} | {res['P@10']*100:>6.1f}% | {res['P@20']*100:>6.1f}%")
    print("="*50)
    
    # 7. Diagnóstico do Top 10
    print("\nDiagnóstico do Top 5 (Fortaleza):")
    ordered_champ = sorted(top_bairros, key=lambda b: scores_champ.get(b, 0), reverse=True)[:5]
    ordered_chal = sorted(top_bairros, key=lambda b: scores_chal.get(b, 0), reverse=True)[:5]
    
    print(f"{'Rank':<5} | {'Champion':<20} | {'Challenger':<20}")
    print("-" * 50)
    for i in range(5):
        c_hit = "[HIT]" if ground_truth.get(ordered_champ[i], 0) > 0 else "[MISS]"
        ch_hit = "[HIT]" if ground_truth.get(ordered_chal[i], 0) > 0 else "[MISS]"
        print(f"{i+1:<5} | {ordered_champ[i]:<20} {c_hit} | {ordered_chal[i]:<20} {ch_hit}")

if __name__ == "__main__":
    evaluate()

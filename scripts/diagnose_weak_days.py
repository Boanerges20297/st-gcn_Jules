#!/usr/bin/env python
"""
diagnose_weak_days.py

Diagnosticar por que Quarta (P@5=0.4) e Sexta (P@5=0.4) têm baixa performance
enquanto Quinta (P@5=1.0) e Sábado (P@5=1.0) são excelentes
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_data():
    """Carrega dados"""
    pkl_path = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def analyze_day_patterns():
    """Analisa padrões de eventos por dia da semana"""
    
    print("=" * 80)
    print("DIAGNÓSTICO DE DIAS FRACOS (QUARTA E SEXTA)")
    print("=" * 80)
    
    data = load_data()
    node_features = data['node_features']  # (319, N_days, 26)
    dates = data['dates']
    
    cvli_data = node_features[:, :, 0]  # (319, N_days)
    num_nodes, num_days = cvli_data.shape
    
    print(f"\nTotal de nós: {num_nodes}")
    print(f"Total de dias: {num_days}")
    print(f"Período: {dates[0]} até {dates[-1]}")
    
    # Separar últimos 30 dias (teste) vs resto (treino)
    last_date = dates[-1]
    cutoff_date = last_date - timedelta(days=30)
    test_start_idx = next((i for i, d in enumerate(dates) if d >= cutoff_date), num_days - 30)
    
    print(f"\nSplit treino/teste:")
    print(f"  Treino: {dates[0]} até {dates[test_start_idx-1]}")
    print(f"  Teste: {dates[test_start_idx]} até {dates[-1]}")
    
    # Análise por dia da semana
    day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    stats_by_day = {}
    
    for day_num in range(7):
        day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
        train_indices = [i for i in day_indices if i < test_start_idx]
        test_indices = [i for i in day_indices if i >= test_start_idx]
        
        # Estatísticas de treino
        train_data = cvli_data[:, train_indices]  # (319, N_train_days)
        train_totals = train_data.sum(axis=0)  # Total de eventos por dia
        train_active_nodes = (train_data > 0).sum(axis=0)  # Nós ativos por dia
        
        # Estatísticas de teste
        test_data = cvli_data[:, test_indices]  # (319, N_test_days)
        test_totals = test_data.sum(axis=0)
        test_active_nodes = (test_data > 0).sum(axis=0)
        
        stats_by_day[day_num] = {
            'name': day_names[day_num],
            'n_train': len(train_indices),
            'n_test': len(test_indices),
            # Treino
            'train_total_mean': train_totals.mean(),
            'train_total_std': train_totals.std(),
            'train_nodes_mean': train_active_nodes.mean(),
            'train_nodes_std': train_active_nodes.std(),
            'train_sparsity': 1 - (train_data > 0).sum() / train_data.size,
            # Teste
            'test_total_mean': test_totals.mean(),
            'test_total_std': test_totals.std(),
            'test_nodes_mean': test_active_nodes.mean(),
            'test_nodes_std': test_active_nodes.std(),
            'test_sparsity': 1 - (test_data > 0).sum() / test_data.size,
            # Variabilidade
            'train_cv': train_totals.std() / (train_totals.mean() + 1e-6),
            'test_cv': test_totals.std() / (test_totals.mean() + 1e-6),
        }
    
    print("\n" + "=" * 80)
    print("ESTATÍSTICAS POR DIA DA SEMANA")
    print("=" * 80)
    
    # DataFrame para análise
    df_stats = pd.DataFrame(stats_by_day).T
    
    print("\n1. VOLUME DE EVENTOS (Treino)")
    print("-" * 60)
    for day_num in range(7):
        s = stats_by_day[day_num]
        print(f"{s['name']:<10} | N={s['n_train']:3d} | "
              f"Eventos/dia: {s['train_total_mean']:5.2f} ± {s['train_total_std']:4.2f} | "
              f"Nós ativos: {s['train_nodes_mean']:4.1f} ± {s['train_nodes_std']:3.1f}")
    
    print("\n2. VOLUME DE EVENTOS (Teste - últimos 30 dias)")
    print("-" * 60)
    for day_num in range(7):
        s = stats_by_day[day_num]
        print(f"{s['name']:<10} | N={s['n_test']:3d} | "
              f"Eventos/dia: {s['test_total_mean']:5.2f} ± {s['test_total_std']:4.2f} | "
              f"Nós ativos: {s['test_nodes_mean']:4.1f} ± {s['test_nodes_std']:3.1f}")
    
    print("\n3. ESPARSIDADE (% de zeros)")
    print("-" * 60)
    for day_num in range(7):
        s = stats_by_day[day_num]
        print(f"{s['name']:<10} | Treino: {s['train_sparsity']*100:5.2f}% | "
              f"Teste: {s['test_sparsity']*100:5.2f}%")
    
    print("\n4. VARIABILIDADE (Coeficiente de Variação)")
    print("-" * 60)
    for day_num in range(7):
        s = stats_by_day[day_num]
        print(f"{s['name']:<10} | Treino CV: {s['train_cv']:5.3f} | "
              f"Teste CV: {s['test_cv']:5.3f}")
    
    # Comparar dias fracos vs fortes
    print("\n" + "=" * 80)
    print("COMPARAÇÃO DIAS FRACOS vs FORTES")
    print("=" * 80)
    
    weak_days = [2, 4]  # Quarta (2), Sexta (4)
    strong_days = [3, 5]  # Quinta (3), Sábado (5)
    
    print("\nDIAS FRACOS (Quarta P@5=0.4, Sexta P@5=0.4)")
    print("-" * 60)
    for day_num in weak_days:
        s = stats_by_day[day_num]
        print(f"\n{s['name']}:")
        print(f"  Treino: {s['n_train']} dias, {s['train_total_mean']:.2f} eventos/dia")
        print(f"  Teste: {s['n_test']} dias, {s['test_total_mean']:.2f} eventos/dia")
        print(f"  Esparsidade treino: {s['train_sparsity']*100:.2f}%")
        print(f"  Variabilidade: CV={s['train_cv']:.3f}")
    
    print("\nDIAS FORTES (Quinta P@5=1.0, Sábado P@5=1.0)")
    print("-" * 60)
    for day_num in strong_days:
        s = stats_by_day[day_num]
        print(f"\n{s['name']}:")
        print(f"  Treino: {s['n_train']} dias, {s['train_total_mean']:.2f} eventos/dia")
        print(f"  Teste: {s['n_test']} dias, {s['test_total_mean']:.2f} eventos/dia")
        print(f"  Esparsidade treino: {s['train_sparsity']*100:.2f}%")
        print(f"  Variabilidade: CV={s['train_cv']:.3f}")
    
    # Análise de distribuição espacial
    print("\n" + "=" * 80)
    print("ANÁLISE DE CONCENTRAÇÃO ESPACIAL")
    print("=" * 80)
    
    for day_num in range(7):
        day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
        train_indices = [i for i in day_indices if i < test_start_idx]
        
        if len(train_indices) == 0:
            continue
        
        train_data = cvli_data[:, train_indices]
        node_totals = train_data.sum(axis=1)  # Total por nó ao longo dos dias de treino
        
        # Top-10 nós
        top10_indices = np.argsort(-node_totals)[:10]
        top10_total = node_totals[top10_indices].sum()
        all_total = node_totals.sum()
        top10_concentration = (top10_total / all_total * 100) if all_total > 0 else 0
        
        # Gini coefficient (desigualdade)
        sorted_totals = np.sort(node_totals[node_totals > 0])
        n = len(sorted_totals)
        if n > 0:
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_totals)) / (n * np.sum(sorted_totals)) - (n + 1) / n
        else:
            gini = 0
        
        s = stats_by_day[day_num]
        s['top10_concentration'] = top10_concentration
        s['gini'] = gini
        s['active_nodes_total'] = (node_totals > 0).sum()
    
    print("\nConcentração de eventos (% nos Top-10 nós):")
    print("-" * 60)
    for day_num in range(7):
        s = stats_by_day[day_num]
        marker = "⚠️" if day_num in weak_days else "✅" if day_num in strong_days else ""
        print(f"{s['name']:<10} | Top-10: {s['top10_concentration']:5.2f}% | "
              f"Gini: {s['gini']:.3f} | Nós ativos: {s['active_nodes_total']:3d} {marker}")
    
    # Análise de consistência temporal
    print("\n" + "=" * 80)
    print("ANÁLISE DE CONSISTÊNCIA TEMPORAL (Top-5 nós)")
    print("=" * 80)
    
    for day_num in [2, 3, 4, 5]:  # Quarta, Quinta, Sexta, Sábado
        day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
        train_indices = [i for i in day_indices if i < test_start_idx]
        test_indices = [i for i in day_indices if i >= test_start_idx]
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
        
        train_data = cvli_data[:, train_indices]
        test_data = cvli_data[:, test_indices]
        
        # Top-5 nós no treino
        train_node_totals = train_data.sum(axis=1)
        top5_train = np.argsort(-train_node_totals)[:5]
        
        # Top-5 nós no teste
        test_node_totals = test_data.sum(axis=1)
        top5_test = np.argsort(-test_node_totals)[:5]
        
        # Overlap
        overlap = len(set(top5_train) & set(top5_test))
        
        s = stats_by_day[day_num]
        marker = "⚠️ FRACO" if day_num in weak_days else "✅ FORTE" if day_num in strong_days else ""
        
        print(f"\n{s['name']} {marker}:")
        print(f"  Top-5 treino: {top5_train.tolist()}")
        print(f"  Top-5 teste:  {top5_test.tolist()}")
        print(f"  Overlap: {overlap}/5 ({overlap/5*100:.0f}%)")
        
        if overlap < 3:
            print(f"  ⚠️ BAIXA CONSISTÊNCIA - Top-5 mudou muito entre treino/teste!")
    
    # Diagnóstico final
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO FINAL")
    print("=" * 80)
    
    print("\nPOSSÍVEIS CAUSAS DE P@5 BAIXO (Quarta e Sexta):")
    print("-" * 60)
    
    # Verificar diferenças
    weak_avg_events = np.mean([stats_by_day[d]['train_total_mean'] for d in weak_days])
    strong_avg_events = np.mean([stats_by_day[d]['train_total_mean'] for d in strong_days])
    
    weak_avg_cv = np.mean([stats_by_day[d]['train_cv'] for d in weak_days])
    strong_avg_cv = np.mean([stats_by_day[d]['train_cv'] for d in strong_days])
    
    weak_avg_concentration = np.mean([stats_by_day[d]['top10_concentration'] for d in weak_days])
    strong_avg_concentration = np.mean([stats_by_day[d]['top10_concentration'] for d in strong_days])
    
    print(f"\n1. Volume de eventos:")
    print(f"   Dias fracos: {weak_avg_events:.2f} eventos/dia")
    print(f"   Dias fortes: {strong_avg_events:.2f} eventos/dia")
    if weak_avg_events < strong_avg_events * 0.8:
        print(f"   ⚠️ Dias fracos têm {(1 - weak_avg_events/strong_avg_events)*100:.1f}% menos eventos")
    
    print(f"\n2. Variabilidade:")
    print(f"   Dias fracos: CV={weak_avg_cv:.3f}")
    print(f"   Dias fortes: CV={strong_avg_cv:.3f}")
    if weak_avg_cv > strong_avg_cv * 1.2:
        print(f"   ⚠️ Dias fracos são {(weak_avg_cv/strong_avg_cv - 1)*100:.1f}% mais variáveis")
    
    print(f"\n3. Concentração espacial:")
    print(f"   Dias fracos: {weak_avg_concentration:.2f}% nos Top-10")
    print(f"   Dias fortes: {strong_avg_concentration:.2f}% nos Top-10")
    if weak_avg_concentration < strong_avg_concentration * 0.9:
        print(f"   ⚠️ Dias fracos têm eventos mais dispersos")
    
    # Salvar análise
    output_path = os.path.join(ROOT, 'reports', 'diagnose_weak_days.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        # Converter numpy types para tipos nativos
        stats_serializable = {}
        for day_num, s in stats_by_day.items():
            stats_serializable[s['name']] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in s.items()
            }
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\n✓ Análise salva em: {output_path}")
    
    return stats_by_day

if __name__ == "__main__":
    stats = analyze_day_patterns()

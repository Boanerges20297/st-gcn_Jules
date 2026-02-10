#!/usr/bin/env python
"""
analyze_ranking_predictions.py

Analisa as predições do modelo de ranking para cada dia da semana
para entender por que Quarta e Sexta têm P@5 baixo
"""

import os
import sys
import pickle
import numpy as np
import torch
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_ranking_final_production import extract_features_enhanced

def load_data():
    """Carrega dados"""
    pkl_path = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def load_ranking_model(day_num):
    """Carrega modelo de ranking para um dia específico"""
    model_path = os.path.join(ROOT, 'models', 'ranking_by_day', f'ranking_model_day{day_num}.pth')
    
    if not os.path.exists(model_path):
        return None, None, None
    
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    from src.train_ranking_final_production import RankingModelProduction
    
    input_dim = data['config']['input_dim']
    model = RankingModelProduction(input_dim)
    
    # Converter chaves de 'net.' para 'fc.'
    converted_state = {}
    for key, value in data['model_state'].items():
        new_key = key.replace('net.', 'fc.')
        converted_state[new_key] = value
    
    model.load_state_dict(converted_state)
    model.eval()
    
    scaler_mean = data['scaler_mean']
    scaler_scale = data['scaler_scale']
    
    return model, scaler_mean, scaler_scale

def analyze_predictions_for_day(day_num, day_name):
    """Analisa predições do modelo para um dia específico"""
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE PREDIÇÕES: {day_name.upper()}")
    print(f"{'='*80}")
    
    # Carregar dados
    data = load_data()
    node_features = data['node_features']
    dates = data['dates']
    
    cvli_data = node_features[:, :, 0]
    num_nodes, num_days = cvli_data.shape
    
    # Separar treino/teste
    last_date = dates[-1]
    cutoff_date = last_date - timedelta(days=30)
    test_start_idx = next((i for i, d in enumerate(dates) if d >= cutoff_date), num_days - 30)
    
    # Índices deste dia específico
    day_indices = [i for i, d in enumerate(dates) if d.weekday() == day_num]
    test_indices = [i for i in day_indices if i >= test_start_idx]
    
    if len(test_indices) == 0:
        print(f"⚠️ Nenhum dado de teste para {day_name}")
        return
    
    print(f"\nDados de teste: {len(test_indices)} dias ({day_name}s)")
    
    # Carregar modelo
    model, scaler_mean, scaler_scale = load_ranking_model(day_num)
    
    if model is None:
        print(f"⚠️ Modelo não encontrado para {day_name}")
        return
    
    print(f"✓ Modelo carregado")
    
    # Para cada dia de teste
    results = []
    
    for test_idx in test_indices:
        test_date = dates[test_idx]
        
        # Janela de 30 dias ANTES deste dia de teste
        window_start = max(0, test_idx - 30)
        X_window = cvli_data[:, window_start:test_idx]
        
        if X_window.shape[1] < 5:
            continue  # Pular se janela muito pequena
        
        # Ground truth (alvo) = média de CVLI na janela
        y_true = X_window.mean(axis=1)
        
        # Extrair features
        X_features = extract_features_enhanced(X_window)
        
        # Normalizar
        X_scaled = (X_features - scaler_mean) / scaler_scale
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predição
        with torch.no_grad():
            y_pred = model(X_tensor).numpy()
        
        # Top-5 real vs predito
        top5_true = np.argsort(-y_true)[:5]
        top5_pred = np.argsort(-y_pred)[:5]
        
        overlap = len(set(top5_true) & set(top5_pred))
        
        # Analisar erros
        # Quais nós o modelo previu como top-5 mas não eram?
        false_positives = [n for n in top5_pred if n not in top5_true]
        # Quais nós eram top-5 real mas modelo não previu?
        false_negatives = [n for n in top5_true if n not in top5_pred]
        
        result = {
            'date': test_date,
            'overlap': overlap,
            'top5_true': top5_true,
            'top5_pred': top5_pred,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'y_true': y_true,
            'y_pred': y_pred,
        }
        results.append(result)
        
        print(f"\n  {test_date.strftime('%Y-%m-%d')} ({day_name})")
        print(f"    Top-5 real:   {top5_true.tolist()}")
        print(f"    Top-5 pred:   {top5_pred.tolist()}")
        print(f"    Overlap: {overlap}/5 ({overlap/5*100:.0f}%)")
        
        if overlap < 3:
            print(f"    ⚠️ Baixo overlap!")
            if false_positives:
                print(f"    Falsos positivos (previu mas não era): {false_positives}")
                # Analisar por que previu errado
                for fp in false_positives[:2]:  # Top-2 FP
                    print(f"      Nó {fp}: y_true={y_true[fp]:.3f}, y_pred={y_pred[fp]:.3f}")
            if false_negatives:
                print(f"    Falsos negativos (era mas não previu): {false_negatives}")
                for fn in false_negatives[:2]:  # Top-2 FN
                    print(f"      Nó {fn}: y_true={y_true[fn]:.3f}, y_pred={y_pred[fn]:.3f}")
    
    # Estatísticas gerais
    if results:
        avg_overlap = np.mean([r['overlap'] for r in results])
        print(f"\n{'='*60}")
        print(f"RESUMO {day_name.upper()}")
        print(f"{'='*60}")
        print(f"Overlap médio: {avg_overlap:.2f}/5 ({avg_overlap/5*100:.0f}%)")
        
        # Análise de features mais importantes
        print(f"\nAnalysando padrões de erro...")
        
        # Agregar todos os falsos positivos
        all_fp = []
        all_fn = []
        for r in results:
            all_fp.extend(r['false_positives'])
            all_fn.extend(r['false_negatives'])
        
        from collections import Counter
        fp_counts = Counter(all_fp)
        fn_counts = Counter(all_fn)
        
        if fp_counts:
            print(f"\nNós frequentemente SOBRE-previstos (modelo pensa que são top-5):")
            for node, count in fp_counts.most_common(5):
                print(f"  Nó {node}: {count} vezes")
        
        if fn_counts:
            print(f"\nNós frequentemente SUB-previstos (modelo não captura):")
            for node, count in fn_counts.most_common(5):
                print(f"  Nó {node}: {count} vezes")

def main():
    print("=" * 80)
    print("ANÁLISE DETALHADA DE PREDIÇÕES DO RANKING")
    print("=" * 80)
    
    # Focar nos dias problemáticos
    days_to_analyze = [
        (2, 'Quarta', 'FRACO'),
        (3, 'Quinta', 'FORTE'),
        (4, 'Sexta', 'FRACO'),
        (5, 'Sábado', 'FORTE'),
    ]
    
    for day_num, day_name, status in days_to_analyze:
        analyze_predictions_for_day(day_num, f"{day_name} ({status})")
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 80)

if __name__ == "__main__":
    main()

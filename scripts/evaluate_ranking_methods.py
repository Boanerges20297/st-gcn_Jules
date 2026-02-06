#!/usr/bin/env python
"""
evaluate_ranking_methods.py

Avaliação comparativa entre RankingCorrectionSystem e RankingInference
Compara performance, concordância e métricas de ranking
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ranking_correction_system import RankingCorrectionSystem, get_ranking_system
from src.ranking_inference import RankingInference

def calculate_precision_at_k(predicted, actual, k=5):
    """Precision@K"""
    if len(actual) == 0:
        return 0.0
    predicted_k = set(predicted[:k])
    actual_set = set(actual[:k])
    return len(predicted_k & actual_set) / k

def calculate_ndcg_at_k(predicted, actual, k=5):
    """NDCG@K"""
    dcg = 0.0
    for i, node in enumerate(predicted[:k]):
        if node in actual[:k]:
            rank = i + 1
            dcg += 1.0 / np.log2(rank + 1)
    
    # IDCG (ideal)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(actual)))])
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_overlap(list1, list2, k=5):
    """Overlap entre duas listas"""
    return len(set(list1[:k]) & set(list2[:k]))

def load_test_data(window_days=30):
    """Carrega dados de teste dos últimos 30 dias"""
    print("\n[1/6] Carregando dados de teste...")
    
    # Carregar dados processados
    data_path = Path(ROOT) / 'data' / 'processed' / 'cvli_producao.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Dados não encontrados: {data_path}")
    
    df = pd.read_csv(data_path)
    df['data'] = pd.to_datetime(df['data'])
    
    # Últimos 30 dias
    end_date = df['data'].max()
    start_date = end_date - timedelta(days=window_days)
    
    df_test = df[(df['data'] >= start_date) & (df['data'] <= end_date)].copy()
    
    # Contar CVLI por bairro e data
    cvli_counts = df_test.groupby(['bairro_assigned', 'data']).size().reset_index(name='cvli_count')
    
    # Pivot para ter (num_nodes, num_days, cvli)
    cvli_pivot = cvli_counts.pivot_table(
        index='bairro_assigned',
        columns='data',
        values='cvli_count',
        aggfunc='sum',
        fill_value=0
    )
    
    print(f"   Período: {start_date.date()} a {end_date.date()}")
    print(f"   Nós (bairros): {len(cvli_pivot)}, Dias: {len(cvli_pivot.columns)}")
    print(f"   Total eventos: {df_test.shape[0]}")
    
    return cvli_pivot.values, cvli_pivot.index.tolist(), end_date

def load_stgcn_model():
    """Carrega modelo ST-GCN"""
    print("\n[2/6] Carregando modelo ST-GCN...")
    
    model_path = Path(ROOT) / 'models' / 'stgcn_model_v2.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo ST-GCN não encontrado: {model_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Carregar modelo (simplificado - assumindo 319 nós)
    # Na prática, você precisaria carregar o modelo completo com arquitetura correta
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"   Modelo carregado em {device}")
    return checkpoint, device

def get_stgcn_predictions(cvli_data, num_nodes=319):
    """
    Simula predições ST-GCN baseado em agregação de CVLI
    (Simplificado para avaliação - em produção usa o modelo real)
    """
    print("\n[3/6] Gerando predições ST-GCN...")
    
    # Score simples: média móvel exponencial dos últimos 7 dias
    weights = np.exp(np.linspace(-1, 0, min(7, cvli_data.shape[1])))
    weights = weights / weights.sum()
    
    recent_data = cvli_data[:, -len(weights):]
    stgcn_scores = (recent_data * weights).sum(axis=1)
    
    # Adicionar ruído para simular incerteza do modelo
    noise = np.random.normal(0, stgcn_scores.std() * 0.1, size=stgcn_scores.shape)
    stgcn_scores = stgcn_scores + noise
    
    top_indices = np.argsort(-stgcn_scores)
    
    print(f"   Top-5 ST-GCN: {top_indices[:5].tolist()}")
    print(f"   Scores: {stgcn_scores[top_indices[:5]].tolist()}")
    
    return stgcn_scores, top_indices

def evaluate_ranking_correction(stgcn_scores, stgcn_top_indices, cvli_data, day_of_week):
    """Avalia RankingCorrectionSystem"""
    print("\n[4/6] Avaliando RankingCorrectionSystem...")
    
    ranking_system = get_ranking_system()
    
    # Obter scores do ranking
    ranking_scores, confidence = ranking_system.get_ranking_scores(
        cvli_data,
        day_of_week=day_of_week
    )
    
    # Corrigir predição ST-GCN
    corrected_top5, conf, was_corrected = ranking_system.correct_stgcn_prediction(
        stgcn_top_indices[:5],
        cvli_data,
        day_of_week=day_of_week,
        confidence_threshold=0.6
    )
    
    ranking_top_indices = np.argsort(-ranking_scores)
    
    print(f"   Confiança: {confidence:.3f}")
    print(f"   Foi corrigido: {was_corrected}")
    print(f"   Top-5 Corrigido: {corrected_top5}")
    
    return {
        'method': 'RankingCorrectionSystem',
        'top5': corrected_top5,
        'scores': ranking_scores,
        'confidence': confidence,
        'was_corrected': was_corrected,
        'ranking_top5': ranking_top_indices[:5].tolist()
    }

def evaluate_ranking_inference(stgcn_scores, cvli_data, day_of_week):
    """Avalia RankingInference"""
    print("\n[5/6] Avaliando RankingInference...")
    
    # Selecionar modelo do dia
    model_path = Path(ROOT) / 'models' / 'ranking_by_day' / f'ranking_model_day{day_of_week}.pth'
    
    if not model_path.exists():
        print(f"   [WARNING] Modelo não encontrado: {model_path}")
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranking_validator = RankingInference(str(model_path), device=device)
    
    if ranking_validator.model is None:
        print("   [ERROR] Falha ao carregar RankingInference")
        return None
    
    # Extrair features (simplificado - 12 dimensões)
    # Na prática, extract_features_clean() seria usado
    num_nodes = cvli_data.shape[0]
    features = np.random.randn(num_nodes, 12)  # Placeholder
    
    # Para usar o método real, precisaria implementar extract_features_clean
    try:
        combined_scores, top_indices = ranking_validator.validate_stgcn_predictions(
            stgcn_scores,
            features,
            top_k=5
        )
        
        print(f"   Top-5 Combinado (70/30): {top_indices.tolist()}")
        print(f"   Scores combinados: {combined_scores[top_indices].tolist()}")
        
        return {
            'method': 'RankingInference',
            'top5': top_indices.tolist(),
            'scores': combined_scores,
            'combined_weight': '70% ST-GCN + 30% Ranking'
        }
    except Exception as e:
        print(f"   [ERROR] Falha na inferência: {e}")
        return None

def compare_methods(ground_truth_top5, correction_result, inference_result, stgcn_top5):
    """Compara os dois métodos"""
    print("\n[6/6] Comparando métodos...")
    
    # Converter arrays numpy para listas
    if isinstance(stgcn_top5, np.ndarray):
        stgcn_top5 = stgcn_top5.tolist()
    if isinstance(ground_truth_top5, np.ndarray):
        ground_truth_top5 = ground_truth_top5.tolist()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'ground_truth': ground_truth_top5,
        'stgcn_baseline': {
            'top5': stgcn_top5,
            'p5': float(calculate_precision_at_k(stgcn_top5, ground_truth_top5)),
            'ndcg5': float(calculate_ndcg_at_k(stgcn_top5, ground_truth_top5)),
            'overlap_with_truth': int(calculate_overlap(stgcn_top5, ground_truth_top5))
        }
    }
    
    # RankingCorrectionSystem
    if correction_result:
        corr_top5 = correction_result['top5']
        if isinstance(corr_top5, np.ndarray):
            corr_top5 = corr_top5.tolist()
        
        results['correction_system'] = {
            'top5': corr_top5,
            'p5': float(calculate_precision_at_k(corr_top5, ground_truth_top5)),
            'ndcg5': float(calculate_ndcg_at_k(corr_top5, ground_truth_top5)),
            'overlap_with_truth': int(calculate_overlap(corr_top5, ground_truth_top5)),
            'overlap_with_stgcn': int(calculate_overlap(corr_top5, stgcn_top5)),
            'was_corrected': bool(correction_result['was_corrected']),
            'confidence': float(correction_result['confidence'])
        }
    
    # RankingInference
    if inference_result:
        inf_top5 = inference_result['top5']
        if isinstance(inf_top5, np.ndarray):
            inf_top5 = inf_top5.tolist()
        
        results['inference_blend'] = {
            'top5': inf_top5,
            'p5': float(calculate_precision_at_k(inf_top5, ground_truth_top5)),
            'ndcg5': float(calculate_ndcg_at_k(inf_top5, ground_truth_top5)),
            'overlap_with_truth': int(calculate_overlap(inf_top5, ground_truth_top5)),
            'overlap_with_stgcn': int(calculate_overlap(inf_top5, stgcn_top5)),
            'weight': inference_result['combined_weight']
        }
    
    # Comparação direta
    if correction_result and inference_result:
        corr_top5 = correction_result['top5']
        inf_top5 = inference_result['top5']
        if isinstance(corr_top5, np.ndarray):
            corr_top5 = corr_top5.tolist()
        if isinstance(inf_top5, np.ndarray):
            inf_top5 = inf_top5.tolist()
        
        results['direct_comparison'] = {
            'overlap_correction_vs_inference': int(calculate_overlap(corr_top5, inf_top5)),
            'delta_p5': float(
                results['inference_blend']['p5'] - 
                results['correction_system']['p5']
            ),
            'delta_ndcg5': float(
                results['inference_blend']['ndcg5'] - 
                results['correction_system']['ndcg5']
            )
        }
    
    return results

def print_comparison_report(results):
    """Imprime relatório comparativo"""
    print("\n" + "="*70)
    print("RELATÓRIO COMPARATIVO: RankingCorrectionSystem vs RankingInference")
    print("="*70)
    
    print("\n📊 MÉTRICAS DE DESEMPENHO:")
    print("-" * 70)
    
    # ST-GCN Baseline
    baseline = results['stgcn_baseline']
    print(f"\n1️⃣  ST-GCN (Baseline)")
    print(f"    Top-5: {baseline['top5']}")
    print(f"    P@5:   {baseline['p5']:.3f}")
    print(f"    NDCG@5: {baseline['ndcg5']:.3f}")
    
    # RankingCorrectionSystem
    if 'correction_system' in results:
        corr = results['correction_system']
        print(f"\n2️⃣  RankingCorrectionSystem (Correção Discreta 4+1)")
        print(f"    Top-5: {corr['top5']}")
        print(f"    P@5:   {corr['p5']:.3f} ({'+' if corr['p5'] >= baseline['p5'] else ''}{corr['p5'] - baseline['p5']:.3f})")
        print(f"    NDCG@5: {corr['ndcg5']:.3f} ({'+' if corr['ndcg5'] >= baseline['ndcg5'] else ''}{corr['ndcg5'] - baseline['ndcg5']:.3f})")
        print(f"    Overlap com ST-GCN: {corr['overlap_with_stgcn']}/5")
        print(f"    Foi corrigido: {'Sim' if corr['was_corrected'] else 'Não'}")
        print(f"    Confiança: {corr['confidence']:.3f}")
    
    # RankingInference
    if 'inference_blend' in results:
        inf = results['inference_blend']
        print(f"\n3️⃣  RankingInference (Blend Contínuo 70/30)")
        print(f"    Top-5: {inf['top5']}")
        print(f"    P@5:   {inf['p5']:.3f} ({'+' if inf['p5'] >= baseline['p5'] else ''}{inf['p5'] - baseline['p5']:.3f})")
        print(f"    NDCG@5: {inf['ndcg5']:.3f} ({'+' if inf['ndcg5'] >= baseline['ndcg5'] else ''}{inf['ndcg5'] - baseline['ndcg5']:.3f})")
        print(f"    Overlap com ST-GCN: {inf['overlap_with_stgcn']}/5")
        print(f"    Peso: {inf['weight']}")
    
    # Comparação Direta
    if 'direct_comparison' in results:
        comp = results['direct_comparison']
        print(f"\n🔍 COMPARAÇÃO DIRETA:")
        print(f"    Overlap Correção vs Inferência: {comp['overlap_correction_vs_inference']}/5")
        print(f"    Δ P@5 (Inferência - Correção): {comp['delta_p5']:+.3f}")
        print(f"    Δ NDCG@5 (Inferência - Correção): {comp['delta_ndcg5']:+.3f}")
        
        # Vencedor
        if comp['delta_p5'] > 0.05:
            winner = "RankingInference"
        elif comp['delta_p5'] < -0.05:
            winner = "RankingCorrectionSystem"
        else:
            winner = "Empate Técnico"
        
        print(f"\n🏆 VENCEDOR (P@5): {winner}")
    
    print("\n" + "="*70)

def main():
    """Execução principal"""
    print("\n🔬 AVALIAÇÃO COMPARATIVA: MÉTODOS DE RANKING")
    print("=" * 70)
    
    try:
        # 1. Carregar dados
        cvli_data, node_ids, end_date = load_test_data(window_days=30)
        
        # 2. Determinar dia da semana
        day_of_week = end_date.weekday()
        print(f"\nDia da semana: {day_of_week} ({end_date.strftime('%A')})")
        
        # 3. Predições ST-GCN
        stgcn_scores, stgcn_top_indices = get_stgcn_predictions(cvli_data)
        stgcn_top5 = stgcn_top_indices[:5].tolist()
        
        # 4. Ground truth (últimos 3 dias reais)
        ground_truth_scores = cvli_data[:, -3:].sum(axis=1)
        ground_truth_top5 = np.argsort(-ground_truth_scores)[:5].tolist()
        print(f"\nGround Truth Top-5 (últimos 3 dias): {ground_truth_top5}")
        
        # 5. Avaliar RankingCorrectionSystem
        correction_result = evaluate_ranking_correction(
            stgcn_scores,
            stgcn_top_indices,
            cvli_data,
            day_of_week
        )
        
        # 6. Avaliar RankingInference
        inference_result = evaluate_ranking_inference(
            stgcn_scores,
            cvli_data,
            day_of_week
        )
        
        # 7. Comparar
        results = compare_methods(
            ground_truth_top5,
            correction_result,
            inference_result,
            stgcn_top5
        )
        
        # 8. Relatório
        print_comparison_report(results)
        
        # 9. Salvar resultados
        output_path = Path(ROOT) / 'reports' / 'ranking_methods_comparison.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Resultados salvos em: {output_path}")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

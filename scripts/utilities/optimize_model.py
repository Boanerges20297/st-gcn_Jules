"""
Análise e Sugestões de Melhoria para o Modelo ST-GCN
Testa diferentes configurações e estratégias
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_predictions_2025 import ModelTester


class ModelOptimizer:
    """Testa diferentes configurações para otimizar performance"""
    
    def __init__(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.output_dir = os.path.join(self.base_dir, 'reports', 'optimization')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_threshold_optimization(self, predictions, actuals):
        """Testa diferentes thresholds para encontrar o ótimo"""
        print("\n" + "="*80)
        print("OTIMIZAÇÃO DE THRESHOLD")
        print("="*80)
        
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        actual_binary = (actual_flat > 0).astype(int)
        
        # Testa múltiplos thresholds
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        for threshold in thresholds:
            pred_binary = (pred_flat > threshold).astype(int)
            
            tp = ((pred_binary == 1) & (actual_binary == 1)).sum()
            fp = ((pred_binary == 1) & (actual_binary == 0)).sum()
            tn = ((pred_binary == 0) & (actual_binary == 0)).sum()
            fn = ((pred_binary == 0) & (actual_binary == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(actual_binary)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            })
        
        df = pd.DataFrame(results)
        
        print("\n📊 Performance por Threshold:")
        print(df.to_string(index=False))
        
        # Encontra threshold ótimo (maior F1)
        best_idx = df['f1_score'].idxmax()
        best_threshold = df.loc[best_idx, 'threshold']
        
        print(f"\n✅ MELHOR THRESHOLD: {best_threshold}")
        print(f"   F1-Score: {df.loc[best_idx, 'f1_score']:.4f}")
        print(f"   Precisão: {df.loc[best_idx, 'precision']:.4f}")
        print(f"   Recall: {df.loc[best_idx, 'recall']:.4f}")
        
        return df, best_threshold
    
    def test_percentile_calibration(self, predictions, actuals):
        """Usa percentis para calibrar predições"""
        print("\n" + "="*80)
        print("CALIBRAÇÃO POR PERCENTIS")
        print("="*80)
        
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        # Agrupa por percentis
        percentiles = [50, 75, 90, 95, 99]
        
        print("\n📊 Taxa de Crime Real por Percentil de Predição:")
        for p in percentiles:
            threshold = np.percentile(pred_flat, p)
            high_risk = pred_flat >= threshold
            
            total_high_risk = high_risk.sum()
            crimes_in_high_risk = (actual_flat[high_risk] > 0).sum()
            hit_rate = crimes_in_high_risk / total_high_risk if total_high_risk > 0 else 0
            
            print(f"  Top {100-p}% (threshold={threshold:.4f}): "
                  f"{crimes_in_high_risk}/{total_high_risk} = {hit_rate:.2%} taxa de acerto")
        
        return percentiles
    
    def test_precision_at_k(self, predictions, actuals, k_values=[10, 50, 100, 200]):
        """Precision@K - Dos top K nós de maior risco, quantos têm crime?"""
        print("\n" + "="*80)
        print("PRECISION@K - RANKING QUALITY")
        print("="*80)
        
        # Média temporal
        pred_mean = predictions.mean(axis=0)  # (N,)
        actual_sum = actuals.sum(axis=0)      # (N,) - total de crimes por nó
        
        # Ordena nós por predição (maior risco primeiro)
        sorted_indices = np.argsort(pred_mean)[::-1]
        
        print("\n📊 Precision@K (nós com maior risco):")
        results = []
        
        for k in k_values:
            top_k_indices = sorted_indices[:k]
            crimes_in_top_k = (actual_sum[top_k_indices] > 0).sum()
            precision_at_k = crimes_in_top_k / k
            
            print(f"  Top {k:3d} nós: {crimes_in_top_k}/{k} = {precision_at_k:.2%} têm crimes")
            
            results.append({
                'k': k,
                'precision_at_k': precision_at_k,
                'crimes_found': int(crimes_in_top_k)
            })
        
        return pd.DataFrame(results)
    
    def analyze_temporal_degradation(self, predictions, actuals, valid_dates):
        """Verifica se modelo degrada ao longo do tempo"""
        print("\n" + "="*80)
        print("ANÁLISE DE DEGRADAÇÃO TEMPORAL")
        print("="*80)
        
        df = pd.DataFrame({
            'date': valid_dates,
            'mae': [np.abs(predictions[i] - actuals[i]).mean() for i in range(len(predictions))]
        })
        
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_mae = df.groupby('month')['mae'].mean()
        
        print("\n📊 MAE por Mês:")
        for month, mae in monthly_mae.items():
            print(f"  {month}: {mae:.4f}")
        
        # Tendência
        if len(monthly_mae) > 1:
            trend = np.polyfit(range(len(monthly_mae)), monthly_mae.values, 1)[0]
            if trend > 0:
                print(f"\n⚠️  DEGRADAÇÃO: MAE aumentando {trend:.6f} por mês")
            else:
                print(f"\n✅ ESTÁVEL: MAE diminuindo {abs(trend):.6f} por mês")
        
        return monthly_mae
    
    def test_only_2025_data(self):
        """Testa modelo apenas com dados de 2025"""
        print("\n" + "="*80)
        print("TESTE: APENAS DADOS DE 2025")
        print("="*80)
        
        tester = ModelTester()
        tester.load_data()
        tester.load_model()
        
        # Filtra apenas 2025
        test_indices, test_dates = tester.prepare_test_data(
            start_date='2025-01-01',
            end_date='2025-12-31'
        )
        
        predictions, actuals, valid_dates = tester.run_predictions(test_indices)
        metrics = tester.calculate_metrics(predictions, actuals)
        
        return predictions, actuals, valid_dates, metrics
    
    def compare_windows(self):
        """Compara diferentes janelas temporais (limitado pelo modelo atual)"""
        print("\n" + "="*80)
        print("ANÁLISE DE JANELA TEMPORAL")
        print("="*80)
        
        print("\n⚠️  LIMITAÇÃO ATUAL:")
        print("  Modelo foi treinado com window_size=7 dias")
        print("  Para testar outras janelas, seria necessário:")
        print("    1. Retreinar modelo com window_size=14, 21, ou 30")
        print("    2. Reprocessar dados com nova configuração")
        print("    3. Validar performance em holdout set")
        
        print("\n💡 SUGESTÕES DE JANELAS PARA RETREINAMENTO:")
        windows = [7, 14, 21, 30]
        for w in windows:
            if w == 7:
                print(f"  {w:2d} dias (atual) - Captura padrões semanais")
            elif w == 14:
                print(f"  {w:2d} dias - Captura 2 semanas, mais estável")
            elif w == 21:
                print(f"  {w:2d} dias - Captura 3 semanas, padrões mensais parciais")
            elif w == 30:
                print(f"  {w:2d} dias - Captura mês completo, mais contexto")
    
    def generate_recommendations(self, predictions, actuals, valid_dates):
        """Gera recomendações finais"""
        print("\n" + "="*80)
        print("RECOMENDAÇÕES DE MELHORIA")
        print("="*80)
        
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()
        
        # 1. Análise de distribuição
        print("\n📊 DISTRIBUIÇÃO DOS DADOS:")
        print(f"  Predições: min={pred_flat.min():.4f}, max={pred_flat.max():.4f}, "
              f"mean={pred_flat.mean():.4f}, std={pred_flat.std():.4f}")
        print(f"  Reais: min={actual_flat.min():.4f}, max={actual_flat.max():.4f}, "
              f"mean={actual_flat.mean():.4f}, std={actual_flat.std():.4f}")
        
        ratio = pred_flat.mean() / actual_flat.mean() if actual_flat.mean() > 0 else 0
        print(f"\n  Razão pred/real: {ratio:.2f}x (modelo superestima {ratio:.1f} vezes)")
        
        # 2. Sugestões prioritizadas
        print("\n🎯 AÇÕES PRIORITÁRIAS (SEM RETREINAMENTO):")
        print("\n  1. AJUSTAR THRESHOLD")
        print("     ├─ Atual: 0.5 (inadequado para eventos raros)")
        print("     ├─ Testar: 0.01, 0.05, 0.1")
        print("     └─ Usar curva ROC para encontrar ótimo")
        
        print("\n  2. USAR MÉTRICAS DE RANKING")
        print("     ├─ Precision@K ao invés de acurácia binária")
        print("     ├─ Focar nos top 10%, 5%, 1% de maior risco")
        print("     └─ MAP (Mean Average Precision)")
        
        print("\n  3. CALIBRAÇÃO POR PERCENTIL")
        print("     ├─ Top 1% = Alto Risco (99º percentil)")
        print("     ├─ Top 5% = Médio Risco (95º percentil)")
        print("     └─ Resto = Baixo Risco")
        
        print("\n  4. NORMALIZAÇÃO DE SAÍDAS")
        print("     ├─ MinMax scaling: (pred - min) / (max - min)")
        print("     ├─ Z-score: (pred - mean) / std")
        print("     └─ Isotonic regression")
        
        print("\n🔧 AÇÕES PARA RETREINAMENTO:")
        print("\n  5. AUMENTAR JANELA TEMPORAL")
        print("     ├─ Atual: 7 dias")
        print("     ├─ Sugerido: 14-21 dias")
        print("     ├─ Vantagem: Captura padrões sazonais melhores")
        print("     └─ Desvantagem: Mais parâmetros, treino mais lento")
        
        print("\n  6. USAR APENAS DADOS RECENTES (2024-2025)")
        print("     ├─ Atual: 2022-2026 (~4 anos)")
        print("     ├─ Sugerido: 2024-2025 (~2 anos)")
        print("     ├─ Vantagem: Padrões mais atuais, menos drift")
        print("     └─ Desvantagem: Menos dados de treino")
        
        print("\n  7. CLASS BALANCING")
        print("     ├─ Implementar class_weight no loss")
        print("     ├─ Oversampling de dias com crime")
        print("     ├─ Focal Loss para classes desbalanceadas")
        print("     └─ SMOTE para dados sintéticos")
        
        print("\n  8. ENSEMBLE DE MODELOS")
        print("     ├─ Treinar múltiplos modelos com diferentes seeds")
        print("     ├─ Combinar com voting ou stacking")
        print("     └─ Reduz overfitting, aumenta robustez")
        
        print("\n  9. FEATURE ENGINEERING")
        print("     ├─ Adicionar dia da semana, feriados")
        print("     ├─ Rolling statistics (média móvel 7/14/30 dias)")
        print("     ├─ Lag features (crime 1, 2, 3 dias atrás)")
        print("     └─ Interações entre canais")
        
        print("\n  10. VALIDAÇÃO TEMPORAL")
        print("      ├─ Walk-forward validation")
        print("      ├─ Treino: 2022-2024")
        print("      ├─ Validação: 2024 Q4")
        print("      └─ Teste: 2025")
    
    def run_full_analysis(self):
        """Executa análise completa de otimização"""
        print("\n" + "="*80)
        print("ANÁLISE COMPLETA DE OTIMIZAÇÃO")
        print("="*80)
        print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Carrega resultados do teste base
        print("\n1. Executando teste base com dados 2025...")
        tester = ModelTester()
        tester.load_data()
        tester.load_model()
        
        test_indices, test_dates = tester.prepare_test_data(start_date='2025-01-01')
        predictions, actuals, valid_dates = tester.run_predictions(test_indices)
        base_metrics = tester.calculate_metrics(predictions, actuals)
        
        # 2. Otimização de threshold
        print("\n2. Otimizando threshold...")
        threshold_results, best_threshold = self.test_threshold_optimization(predictions, actuals)
        
        # 3. Calibração por percentil
        print("\n3. Testando calibração por percentil...")
        self.test_percentile_calibration(predictions, actuals)
        
        # 4. Precision@K
        print("\n4. Avaliando Precision@K...")
        precision_at_k = self.test_precision_at_k(predictions, actuals)
        
        # 5. Degradação temporal
        print("\n5. Analisando degradação temporal...")
        monthly_mae = self.analyze_temporal_degradation(predictions, actuals, valid_dates)
        
        # 6. Análise de janela temporal
        print("\n6. Analisando janela temporal...")
        self.compare_windows()
        
        # 7. Recomendações finais
        print("\n7. Gerando recomendações...")
        self.generate_recommendations(predictions, actuals, valid_dates)
        
        # Salva resultados
        results = {
            'timestamp': datetime.now().isoformat(),
            'base_metrics': base_metrics,
            'best_threshold': float(best_threshold),
            'threshold_analysis': threshold_results.to_dict('records'),
            'precision_at_k': precision_at_k.to_dict('records'),
            'monthly_mae': {str(k): float(v) for k, v in monthly_mae.items()}
        }
        
        output_file = os.path.join(self.output_dir, 'optimization_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Resultados salvos em: {output_file}")
        
        print("\n" + "="*80)
        print("ANÁLISE CONCLUÍDA")
        print("="*80)
        
        return results


def main():
    """Função principal"""
    optimizer = ModelOptimizer()
    
    try:
        results = optimizer.run_full_analysis()
        return results
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    results = main()

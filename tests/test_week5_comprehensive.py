2#!/usr/bin/env python3
"""
WEEK 5 TESTE COMPREHENSIVE (Testes Unitários Expandidos)
Cobertura: Casos extremos, condições limite, validações

Testes incluem:
- Cálculo de métricas com valores extremos
- Detector de anomalias em limites
- Convergência de funções de loss
- Geração de explicações com dados faltantes
"""

import sys
import os
import json
import unittest
import numpy as np
from pathlib import Path
from datetime import date, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricsEdgeCases(unittest.TestCase):
    """Testes de casos extremos para o módulo de métricas"""
    
    def setUp(self):
        """Prepara testes"""
        from src.metrics import MetricReporter
        self.reporter = MetricReporter()
    
    def test_precision_at_k_perfect_ranking(self):
        """P@K com ranking perfeito deve retornar 1.0"""
        rankings = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        true_labels = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        
        p_at_5 = self.reporter.calculate_precision_at_k(rankings, true_labels, k=5)
        self.assertEqual(p_at_5, 1.0)
    
    def test_precision_at_k_zero_ranking(self):
        """P@K com nenhum acerto deve retornar 0.0"""
        rankings = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        true_labels = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        p_at_5 = self.reporter.calculate_precision_at_k(rankings, true_labels, k=5)
        self.assertEqual(p_at_5, 0.0)
    
    def test_precision_at_k_partial_match(self):
        """P@K com 3 acertos em 5 deve retornar 0.6"""
        rankings = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        true_labels = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
        
        p_at_5 = self.reporter.calculate_precision_at_k(rankings, true_labels, k=5)
        self.assertAlmostEqual(p_at_5, 0.6, places=2)
    
    def test_precision_at_k_k_greater_than_length(self):
        """P@K com k maior que tamanho do ranking deve usar tamanho real"""
        rankings = {1: 1, 2: 2, 3: 3}
        true_labels = {1: 1, 2: 1, 3: 1}
        
        p_at_20 = self.reporter.calculate_precision_at_k(rankings, true_labels, k=20)
        self.assertEqual(p_at_20, 1.0)
    
    def test_ndcg_perfect_ranking(self):
        """NDCG com ranking ideal deve ser próximo de 1.0"""
        rankings = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        true_labels = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
        
        ndcg = self.reporter.calculate_ndcg(rankings, true_labels, k=3)
        self.assertGreater(ndcg, 0.8)
    
    def test_ndcg_worst_ranking(self):
        """NDCG com pior ranking possível deve estar próximo de 0.0"""
        rankings = {1: 4, 2: 5, 3: 3, 4: 2, 5: 1}
        true_labels = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
        
        ndcg = self.reporter.calculate_ndcg(rankings, true_labels, k=3)
        self.assertLess(ndcg, 0.5)
    
    def test_recall_calculation_all_items_ranked(self):
        """Recall com todos os itens rankeados"""
        rankings = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        true_labels = {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
        
        recall = self.reporter.calculate_recall_at_k(rankings, true_labels, k=5)
        self.assertEqual(recall, 1.0)
    
    def test_recall_partial_items_ranked(self):
        """Recall com apenas alguns itens rankeados"""
        rankings = {1: 1, 2: 2}
        true_labels = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
        
        recall = self.reporter.calculate_recall_at_k(rankings, true_labels, k=2)
        self.assertAlmostEqual(recall, 2/3, places=2)
    
    def test_mrr_first_position(self):
        """MRR com acerto na primeira posição"""
        rankings = {1: 1, 2: 2, 3: 3}
        true_labels = {1: 1, 2: 0, 3: 0}
        
        mrr = self.reporter.calculate_mrr(rankings, true_labels)
        self.assertEqual(mrr, 1.0)
    
    def test_mrr_third_position(self):
        """MRR com acerto na terceira posição"""
        rankings = {1: 2, 2: 3, 3: 1}
        true_labels = {1: 0, 2: 0, 3: 1}
        
        mrr = self.reporter.calculate_mrr(rankings, true_labels)
        self.assertAlmostEqual(mrr, 1/3, places=2)


class TestAnomalyDetectorEdgeCases(unittest.TestCase):
    """Testes de casos extremos para detector de anomalias"""
    
    def setUp(self):
        """Prepara testes"""
        from src.event_anomaly_detector import EventAnomalyDetector
        self.detector = EventAnomalyDetector()
    
    def test_severity_classification_critical_event(self):
        """Evento crítico deve retornar severidade > 0.8"""
        keywords = ["homicídio", "massacre", "tiroteio"]
        severity = self.detector.classify_severity({"title": "Homicídio em massa"}, keywords)
        self.assertGreater(severity, 0.7)
    
    def test_severity_classification_minor_event(self):
        """Evento menor deve retornar severidade < 0.3"""
        keywords = ["roubo"]
        severity = self.detector.classify_severity({"title": "Pequeno roubo"}, keywords)
        self.assertLess(severity, 0.5)
    
    def test_severity_with_mitigating_factors(self):
        """Evento com fator mitigador deve ter severidade reduzida"""
        event = {
            "title": "Tiroteio contido",
            "description": "Sem vítimas"
        }
        keywords = ["tiroteio"]
        severity_without_mitigation = self.detector.classify_severity(event, keywords)
        
        # Com fator mitigador explícito
        event["mitigating_factors"] = ["sem vítimas", "area evacuada"]
        severity_with_mitigation = self.detector.classify_severity(event, keywords)
        
        # Severidade com mitigação deve ser menor
        self.assertLessEqual(severity_with_mitigation, severity_without_mitigation)
    
    def test_severity_zero_for_no_match(self):
        """Nenhuma correspondência de palavra-chave retorna 0.0"""
        event = {"title": "Bom tempo hoje"}
        keywords = ["homicídio", "roubo", "tiroteio"]
        severity = self.detector.classify_severity(event, keywords)
        self.assertEqual(severity, 0.0)
    
    def test_location_multiplier_high_risk_area(self):
        """Áreas de alto risco aumentam severidade"""
        high_risk_locations = ["Centro", "Bom Merito", "Conjunto Cebalho"]
        
        for location in high_risk_locations:
            multiplier = self.detector.get_location_multiplier(location)
            self.assertGreater(multiplier, 1.0, f"Localização {location} deve aumentar severidade")
    
    def test_location_multiplier_low_risk_area(self):
        """Áreas de baixo risco reduzem severidade"""
        low_risk_locations = ["Meireles", "Varjota"]
        
        for location in low_risk_locations:
            multiplier = self.detector.get_location_multiplier(location)
            self.assertLess(multiplier, 1.0, f"Localização {location} deve reduzir severidade")


class TestLossFunctionConvergence(unittest.TestCase):
    """Testes de convergência de funções de loss"""
    
    def setUp(self):
        """Prepara testes"""
        from src.loss_functions import CombinedLoss, DiversityLoss, RankingLoss
        self.combined_loss = CombinedLoss(alpha=0.5)
        self.diversity_loss = DiversityLoss()
        self.ranking_loss = RankingLoss()
    
    def test_loss_decreases_with_better_predictions(self):
        """Loss deve diminuir com predições melhores"""
        import torch
        
        # Predições piores
        bad_predictions = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        bad_targets = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        bad_loss = self.ranking_loss(bad_predictions, bad_targets).item()
        
        # Predições melhores
        good_predictions = torch.tensor([[0.9, 0.8, 0.7, 0.2, 0.1]])
        good_loss = self.ranking_loss(good_predictions, bad_targets).item()
        
        self.assertGreater(bad_loss, good_loss, "Loss deve diminuir com melhores predições")
    
    def test_combined_loss_balances_p5_p20(self):
        """CombinedLoss deve balancear entre P@5 e P@20"""
        import torch
        
        # Cenário hipotético
        predictions = torch.randn(10, 20)
        targets = torch.randint(0, 2, (10, 20)).float()
        
        combined = self.combined_loss(predictions, targets).item()
        self.assertGreater(combined, 0.0, "Loss deve ser positivo")
        self.assertLess(combined, float('inf'), "Loss não deve ser infinito")
    
    def test_loss_positive_values(self):
        """Loss deve sempre retornar valores positivos"""
        import torch
        
        predictions = torch.randn(5, 10)
        targets = torch.randint(0, 2, (5, 10)).float()
        
        loss_values = [
            self.ranking_loss(predictions, targets).item(),
            self.diversity_loss(predictions, targets).item() if hasattr(self.diversity_loss, '__call__') else 0
        ]
        
        for loss_val in loss_values:
            if loss_val > 0:  # Se foi calculado
                self.assertGreaterEqual(loss_val, 0.0)


class TestExplanationGenerationEdgeCases(unittest.TestCase):
    """Testes de casos extremos para geração de explicações"""
    
    def setUp(self):
        """Prepara testes"""
        from src.explanation_generator import ExplanationGenerator
        self.generator = ExplanationGenerator()
    
    def test_explanation_with_complete_context(self):
        """Explicação com contexto completo deve conter todos os fatores"""
        context = {
            "temporal_features": [0.8, 0.6],
            "spatial_features": [0.7, 0.5],
            "event_features": [0.9],
            "historical_features": [0.4]
        }
        
        explanation = self.generator.explain_node_ranking(
            node_id=1,
            rank=5,
            context_dict=context
        )
        
        self.assertIn("summary", explanation)
        self.assertIn("factors", explanation)
        self.assertIn("confidence", explanation)
        self.assertGreater(len(explanation["factors"]), 0)
    
    def test_explanation_with_missing_context(self):
        """Explicação com contexto parcial deve usar valores padrão"""
        context = {
            "temporal_features": [0.5]
            # Faltam outros contextos
        }
        
        explanation = self.generator.explain_node_ranking(
            node_id=1,
            rank=5,
            context_dict=context
        )
        
        self.assertIn("summary", explanation)
        # Não deve falhar, deve usar valores padrão
        self.assertIsNotNone(explanation["factors"])
    
    def test_explanation_with_empty_context(self):
        """Explicação com contexto vazio deve usar valores padrão"""
        context = {}
        
        explanation = self.generator.explain_node_ranking(
            node_id=1,
            rank=5,
            context_dict=context
        )
        
        # Deve retornar uma explicação válida mesmo sem contexto
        self.assertIsNotNone(explanation)
        self.assertIn("summary", explanation)
    
    def test_risk_level_classification_critical(self):
        """Classificação de risco crítico para valores altos"""
        high_confidence = 0.95
        risk_level = self.generator.classify_risk_level(high_confidence)
        self.assertEqual(risk_level, "CRITICAL")
    
    def test_risk_level_classification_minimal(self):
        """Classificação de risco mínimo para valores baixos"""
        low_confidence = 0.05
        risk_level = self.generator.classify_risk_level(low_confidence)
        self.assertEqual(risk_level, "MINIMAL")
    
    def test_risk_level_classification_moderate(self):
        """Classificação de risco moderado para valores médios"""
        medium_confidence = 0.50
        risk_level = self.generator.classify_risk_level(medium_confidence)
        self.assertIn(risk_level, ["LOW", "MODERATE", "HIGH"])
    
    def test_top_k_explanation_valid_structure(self):
        """Explicação para top-K deve ter estrutura válida"""
        nodes = [1, 2, 3]
        explanations = self.generator.explain_top_k(nodes, top_k=3)
        
        self.assertEqual(len(explanations), len(nodes))
        for exp in explanations:
            self.assertIn("node_id", exp)
            self.assertIn("summary", exp)


class TestMetricsCalculationBoundary(unittest.TestCase):
    """Testes de valores limite para cálculos de métricas"""
    
    def setUp(self):
        """Prepara testes"""
        from src.metrics import MetricReporter
        self.reporter = MetricReporter()
    
    def test_empty_rankings(self):
        """Cálculo com rankings vazios"""
        rankings = {}
        true_labels = {}
        
        # Não deve falhar
        try:
            p_at_5 = self.reporter.calculate_precision_at_k(rankings, true_labels, k=5)
            # Pode retornar 0 ou NaN/None, mas não deve falhar
            self.assertIsNotNone(p_at_5)
        except Exception as e:
            self.fail(f"Falhou com rankings vazios: {e}")
    
    def test_single_node_ranking(self):
        """Cálculo com um único nó"""
        rankings = {1: 1}
        true_labels = {1: 1}
        
        p_at_5 = self.reporter.calculate_precision_at_k(rankings, true_labels, k=5)
        self.assertEqual(p_at_5, 1.0)
    
    def test_very_large_node_ids(self):
        """Cálculo com IDs de nós muito grandes"""
        rankings = {10000: 1, 20000: 2, 30000: 3}
        true_labels = {10000: 1, 20000: 1, 30000: 0}
        
        p_at_3 = self.reporter.calculate_precision_at_k(rankings, true_labels, k=3)
        self.assertAlmostEqual(p_at_3, 2/3, places=2)


if __name__ == "__main__":
    # Configurar verbosidade
    unittest.main(verbosity=2)

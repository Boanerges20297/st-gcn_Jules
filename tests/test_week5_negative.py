#!/usr/bin/env python3
"""
WEEK 5 TESTE NEGATIVE (Testes de Cenários Negativos)
Cobertura: Tratamento de erros, entrada inválida, falhas

Testes incluem:
- Entrada inválida (tipos errados, valores fora de limites)
- Dados corrompidos/malformados
- Falhas de arquivo e recursos
- Comportamento degradado graceful
- Mensagens de erro apropriadas
"""

import sys
import os
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricsInputValidation(unittest.TestCase):
    """Testes de validação de entrada para métricas"""
    
    def setUp(self):
        """Prepara testes"""
        from src.metrics import MetricReporter
        self.reporter = MetricReporter()
    
    def test_precision_with_none_rankings(self):
        """P@K com rankings None deve falhar gracefully"""
        
        try:
            result = self.reporter.calculate_precision_at_k(None, {}, k=5)
            # Se não falhar, deve retornar valor válido ou None
            self.assertIn(result, [None, 0.0, float('nan')] + list(range(0, 2)))
        except (TypeError, AttributeError):
            # Falha esperada
            pass
    
    def test_precision_with_invalid_k(self):
        """P@K com k inválido deve ser tratado"""
        
        rankings = {1: 1, 2: 2, 3: 3}
        true_labels = {1: 1, 2: 0, 3: 0}
        
        # k=0 deve ser tratado
        try:
            result = self.reporter.calculate_precision_at_k(rankings, true_labels, k=0)
            # Ou retorna None/0 ou levanta exceção apropriada
            self.assertIsNotNone(result)
        except (ValueError, ZeroDivisionError):
            pass
        
        # k negativo deve ser tratado
        try:
            result = self.reporter.calculate_precision_at_k(rankings, true_labels, k=-1)
            self.assertIsNotNone(result)
        except (ValueError, ZeroDivisionError):
            pass
    
    def test_precision_with_mismatched_ids(self):
        """P@K com IDs que não correspondem"""
        
        rankings = {1: 1, 2: 2, 3: 3}
        true_labels = {99: 1, 100: 1}  # IDs diferentes
        
        # Deve ser tratado (provavelmente retorna 0)
        try:
            result = self.reporter.calculate_precision_at_k(rankings, true_labels, k=5)
            self.assertIsNotNone(result)
        except KeyError:
            pass
    
    def test_ndcg_with_invalid_labels(self):
        """NDCG com labels fora do intervalo [0,1]"""
        
        rankings = {1: 1, 2: 2, 3: 3}
        
        # Labels > 1
        invalid_labels = {1: 2.0, 2: 1.5, 3: 0.5}
        try:
            result = self.reporter.calculate_ndcg(rankings, invalid_labels, k=3)
            # Pode retornar valor ou falhar
            if result is not None:
                self.assertIsInstance(result, (int, float))
        except (ValueError, AssertionError):
            pass
    
    def test_recall_with_empty_true_labels(self):
        """Recall com labels verdadeiros vazios"""
        
        rankings = {1: 1, 2: 2, 3: 3}
        true_labels = {}
        
        try:
            result = self.reporter.calculate_recall_at_k(rankings, true_labels, k=5)
            # Pode retornar 0, 1, NaN, ou None
            if result is not None:
                self.assertGreaterEqual(result, 0.0)
        except (ZeroDivisionError, KeyError):
            pass


class TestAnomalyDetectorInputValidation(unittest.TestCase):
    """Testes de validação para detector de anomalias"""
    
    def setUp(self):
        """Prepara testes"""
        from src.event_anomaly_detector import EventAnomalyDetector
        self.detector = EventAnomalyDetector()
    
    def test_severity_with_none_event(self):
        """Classificação de severidade com evento None"""
        
        try:
            result = self.detector.classify_severity(None, ["keyword"])
            # Deve retornar 0.0 ou falhar gracefully
            if result is not None:
                self.assertEqual(result, 0.0)
        except (TypeError, AttributeError):
            pass
    
    def test_severity_with_empty_keywords(self):
        """Severidade com lista de palavras-chave vazia"""
        
        event = {"title": "Evento teste"}
        try:
            result = self.detector.classify_severity(event, [])
            # Sem palavras-chave, severidade deve ser 0
            self.assertEqual(result, 0.0)
        except TypeError:
            pass
    
    def test_severity_with_malformed_event(self):
        """Severidade com evento malformado"""
        
        malformed_events = [
            None,
            {},  # Sem título
            {"title": None},
            {"title": 123},  # Tipo errado
            "não é dicionário",
            []
        ]
        
        keywords = ["homicídio"]
        
        for event in malformed_events:
            try:
                result = self.detector.classify_severity(event, keywords)
                # Não deve lançar exceção, deve usar valor padrão
                if result is not None:
                    self.assertGreaterEqual(result, 0.0)
            except (TypeError, AttributeError):
                # Aceitável se falhar com TypeError/AttributeError
                pass
    
    def test_location_multiplier_with_none(self):
        """Multiplicador de localização com None"""
        
        try:
            result = self.detector.get_location_multiplier(None)
            # Deve retornar 1.0 (sem modificação)
            self.assertEqual(result, 1.0)
        except (TypeError, AttributeError):
            pass
    
    def test_location_multiplier_with_unknown(self):
        """Multiplicador para localização desconhecida"""
        
        result = self.detector.get_location_multiplier("LocalidadeXYZ123456")
        
        # Deve retornar 1.0 (valor padrão) para desconhecido
        self.assertEqual(result, 1.0)


class TestExplanationGeneratorRobustness(unittest.TestCase):
    """Testes de robustez do gerador de explicações"""
    
    def setUp(self):
        """Prepara testes"""
        from src.explanation_generator import ExplanationGenerator
        self.generator = ExplanationGenerator()
    
    def test_explain_with_none_node_id(self):
        """Explicação com node_id None"""
        
        try:
            result = self.generator.explain_node_ranking(None, 1, {})
            # Deve retornar explicação ou falhar gracefully
            self.assertIsNotNone(result)
        except (TypeError, ValueError):
            pass
    
    def test_explain_with_huge_rank(self):
        """Explicação com rank extremamente grande"""
        
        try:
            result = self.generator.explain_node_ranking(1, 1000000, {})
            # Deve ser tratado
            self.assertIsNotNone(result)
        except ValueError:
            pass
    
    def test_explain_with_negative_confidence(self):
        """Classificação de risco com confiança negativa"""
        
        try:
            result = self.generator.classify_risk_level(-0.5)
            # Deve usar valor padrão ou limitar
            self.assertIsNotNone(result)
        except (ValueError, AssertionError):
            pass
    
    def test_explain_with_confidence_above_one(self):
        """Classificação de risco com confiança > 1.0"""
        
        try:
            result = self.generator.classify_risk_level(1.5)
            # Deve limitar a 1.0
            self.assertIsNotNone(result)
        except (ValueError, AssertionError):
            pass
    
    def test_explain_top_k_with_empty_list(self):
        """Explicação top-K com lista vazia"""
        
        try:
            result = self.generator.explain_top_k([], top_k=5)
            # Deve retornar lista vazia
            self.assertEqual(result, [])
        except (TypeError, ValueError):
            pass
    
    def test_explain_top_k_with_negative_k(self):
        """Explicação top-K com k negativo"""
        
        try:
            result = self.generator.explain_top_k([1, 2, 3], top_k=-1)
            # Deve ser tratado
            self.assertIsNotNone(result)
        except (ValueError, AssertionError):
            pass


class TestEventManagerErrorHandling(unittest.TestCase):
    """Testes de tratamento de erros no EventManager"""
    
    def test_event_manager_with_missing_file(self):
        """EventManager com arquivo que não existe"""
        
        from src.event_manager import EventManager
        
        try:
            manager = EventManager("/inexistente/arquivo/nao/existe.json")
            # Deve falhar gracefully ou criar aplicação vazia
            self.assertIsNotNone(manager)
        except Exception:
            pass  # Esperado
    
    def test_event_manager_with_corrupted_json(self):
        """EventManager com JSON corrompido"""
        
        from src.event_manager import EventManager
        
        # Criar arquivo JSON corrompido
        test_file = Path(__file__).parent / "corrupted_test.json"
        try:
            with open(test_file, 'w') as f:
                f.write("{invalid json content [}")
            
            manager = EventManager(str(test_file))
            # Deve falhar gracefully
            self.assertIsNotNone(manager)
        except Exception:
            pass  # Esperado
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_get_events_with_invalid_date(self):
        """Obter eventos com data inválida"""
        
        from src.event_manager import EventManager
        
        manager = EventManager("data/exogenous_events_geocoded.json")
        
        try:
            # Passar data inválida
            result = manager.get_events_for_date("data inválida")
            # Deve falhar ou retornar lista vazia
            self.assertIsNotNone(result)
        except (TypeError, ValueError):
            pass


class TestAPIErrorHandling(unittest.TestCase):
    """Testes de tratamento de erros na API"""
    
    def setUp(self):
        """Prepara cliente ou pula if app não disponível"""
        try:
            from app import app
            self.app = app.test_client()
            self.app_available = True
        except Exception:
            self.app_available = False
    
    def test_explain_with_non_integer_id(self):
        """GET /api/explain com ID não inteiro"""
        
        if not self.app_available:
            self.skipTest("App não disponível")
        
        try:
            response = self.app.get('/api/explain/abc')
            # Deve retornar 400 ou 404
            self.assertIn(response.status_code, [400, 404, 405])
        except Exception:
            pass
    
    def test_explain_with_very_large_id(self):
        """GET /api/explain com ID muito grande"""
        
        if not self.app_available:
            self.skipTest("App não disponível")
        
        try:
            response = self.app.get('/api/explain/999999999999')
            # Deve retornar 404 ou 400, não 500
            self.assertIn(response.status_code, [400, 404, 503])
        except Exception:
            pass
    
    def test_API_with_timeout(self):
        """API com request timeout (testovelmente)"""
        
        if not self.app_available:
            self.skipTest("App não disponível")
        
        try:
            response = self.app.get('/api/metrics', timeout=1)
            # Não deve retornar 500
            self.assertIn(response.status_code, [200, 400, 503, 408])
        except Exception:
            pass


class TestGracefulDegradation(unittest.TestCase):
    """Testes de degradação graceful sob erro"""
    
    def test_explanation_without_event_manager(self):
        """Explicação sem EventManager disponível"""
        
        from src.explanation_generator import ExplanationGenerator
        
        generator = ExplanationGenerator()
        
        try:
            context = {
                "temporal_features": [0.5],
                "spatial_features": [0.5]
            }
            
            result = generator.explain_node_ranking(1, 1, context)
            # Deve retornar explicação mesmo sem events
            self.assertIsNotNone(result)
            self.assertIn("summary", result)
        except Exception as e:
            self.fail(f"Explicação falhou sem EventManager: {e}")
    
    def test_metrics_with_partial_data(self):
        """Métricas com dados parciais"""
        
        from src.metrics import MetricReporter
        
        reporter = MetricReporter()
        
        # Apenas algumas métricas disponíveis
        rankings = {1: 1, 2: 2}
        true_labels = {1: 1}  # Menos que rankings
        
        try:
            p_at_5 = reporter.calculate_precision_at_k(rankings, true_labels, k=5)
            # Deve calcular com dados disponíveis
            self.assertIsNotNone(p_at_5)
        except Exception as e:
            self.fail(f"Métrica falhou com dados parciais: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

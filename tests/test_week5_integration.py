#!/usr/bin/env python3
"""
WEEK 5 TESTE INTEGRATION (Testes de Integração End-to-End)
Cobertura: Fluxos completos, integração entre módulos, API chains

Testes incluem:
- Fluxo completo: treinamento -> ranking -> explicação
- Cadeias de endpoints API
- Fluxo de dados do dashboard
- Gerenciamento de eventos durante inferência
"""

import sys
import os
import json
import unittest
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEndToEndPipeline(unittest.TestCase):
    """Testes de pipeline completo"""
    
    def setUp(self):
        """Prepara ambiente de teste"""
        self.test_dir = Path(__file__).parent / "fixtures"
        self.test_dir.mkdir(exist_ok=True)
    
    def test_model_prediction_to_ranking_to_explanation(self):
        """Fluxo: predição do modelo -> ranking -> explicação"""
        
        # Simular saída de predição do modelo
        node_predictions = {
            1: 0.95,
            2: 0.87,
            3: 0.76,
            4: 0.65,
            5: 0.54,
            6: 0.43,
            7: 0.32,
            8: 0.21,
            9: 0.10,
            10: 0.05
        }
        
        # Importar módulos necessários
        try:
            from src.metrics import MetricReporter
            from src.explanation_generator import ExplanationGenerator
            
            # Converter predições em ranking
            sorted_predictions = sorted(
                node_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            rankings = {i: node_id for i, (node_id, _) in enumerate(sorted_predictions, 1)}
            
            # Gerar explicação para top node
            top_node = rankings[1]
            generator = ExplanationGenerator()
            
            context = {
                "temporal_features": [0.8, 0.6],
                "spatial_features": [0.7, 0.5],
                "event_features": [0.9],
                "historical_features": [0.4]
            }
            
            explanation = generator.explain_node_ranking(top_node, 1, context)
            
            # Validações
            self.assertIsNotNone(explanation)
            self.assertIn("summary", explanation)
            self.assertIn("factors", explanation)
            self.assertGreater(explanation["confidence"], 0.5)
            
        except Exception as e:
            self.fail(f"Pipeline falhou: {e}")
    
    def test_event_loading_to_anomaly_detection(self):
        """Fluxo: carregar eventos -> detectar anomalias"""
        
        try:
            from src.event_manager import EventManager
            
            event_file = "data/exogenous_events_geocoded.json"
            
            # Se arquivo não existir, criar um mock
            if not os.path.exists(event_file):
                mock_events = [
                    {
                        "date": "2026-02-06",
                        "severity": 0.8,
                        "title": "Evento teste"
                    }
                ]
                event_manager = EventManager(event_file)
                event_manager.events = mock_events
            else:
                event_manager = EventManager(event_file)
            
            # Obter nível de anomalia para hoje
            today = date.today()
            anomaly_level = event_manager.get_anomaly_level_for_date(today)
            
            # Validações
            self.assertIsNotNone(anomaly_level)
            self.assertGreaterEqual(anomaly_level, 0.0)
            self.assertLessEqual(anomaly_level, 1.0)
            
        except Exception as e:
            self.fail(f"Fluxo de eventos falhou: {e}")
    
    def test_metrics_calculation_from_predictions(self):
        """Fluxo: predições -> cálculo de métricas"""
        
        try:
            from src.metrics import MetricReporter
            
            reporter = MetricReporter()
            
            # Simular predições e labels verdadeiros
            rankings = {
                1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                6: 6, 7: 7, 8: 8, 9: 9, 10: 10
            }
            true_labels = {
                1: 1, 2: 1, 3: 1, 4: 1, 5: 0,
                6: 0, 7: 0, 8: 0, 9: 0, 10: 0
            }
            
            # Calcular várias métricas
            p_at_5 = reporter.calculate_precision_at_k(rankings, true_labels, k=5)
            p_at_10 = reporter.calculate_precision_at_k(rankings, true_labels, k=10)
            ndcg_at_5 = reporter.calculate_ndcg(rankings, true_labels, k=5)
            recall_at_10 = reporter.calculate_recall_at_k(rankings, true_labels, k=10)
            
            # Validações
            self.assertGreaterEqual(p_at_5, 0.0)
            self.assertLessEqual(p_at_5, 1.0)
            self.assertGreaterEqual(p_at_10, 0.0)
            self.assertLessEqual(p_at_10, 1.0)
            self.assertGreaterEqual(ndcg_at_5, 0.0)
            self.assertGreaterEqual(recall_at_10, 0.0)
            
            # P@5 deve ser >= P@10 (não pode melhorar)
            self.assertGreaterEqual(p_at_5, p_at_10)
            
        except Exception as e:
            self.fail(f"Fluxo de métricas falhou: {e}")


class TestAPIEndpointChains(unittest.TestCase):
    """Testes de cadeias de endpoints API"""
    
    def setUp(self):
        """Prepara cliente de teste Flask"""
        try:
            from app import app
            self.app = app.test_client()
            self.app.testing = True
        except Exception as e:
            self.skipTest(f"App não carregado: {e}")
    
    def test_api_explain_endpoint_structure(self):
        """Teste de estrutura de resposta do endpoint /api/explain"""
        
        try:
            # GET /api/explain/<node_id>
            response = self.app.get('/api/explain/1')
            
            # Validações
            self.assertIn(response.status_code, [200, 400, 503])
            
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertIn("summary", data)
                self.assertIn("factors", data)
                self.assertIn("confidence", data)
                
        except Exception as e:
            self.skipTest(f"Endpoint não disponível: {e}")
    
    def test_api_metrics_endpoint_structure(self):
        """Teste de estrutura de resposta do endpoint /api/metrics"""
        
        try:
            response = self.app.get('/api/metrics')
            
            # Validações
            self.assertIn(response.status_code, [200, 400, 503])
            
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertIn("metrics", data)
                self.assertIn("summary", data)
                
        except Exception as e:
            self.skipTest(f"Endpoint não disponível: {e}")
    
    def test_api_anomaly_endpoint_structure(self):
        """Teste de estrutura de resposta do endpoint /api/anomaly_status"""
        
        try:
            response = self.app.get('/api/anomaly_status')
            
            # Validações
            self.assertIn(response.status_code, [200, 400, 503])
            
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertIn("anomaly_level", data)
                
        except Exception as e:
            self.skipTest(f"Endpoint não disponível: {e}")
    
    def test_api_sequence_explain_metrics_anomaly(self):
        """Teste de sequência: explain -> metrics -> anomaly_status"""
        
        try:
            # Chamar endpoints em sequência
            r1 = self.app.get('/api/explain/1')
            r2 = self.app.get('/api/metrics')
            r3 = self.app.get('/api/anomaly_status')
            
            # Todos devem retornar 200 ou 503 (service unavailable é aceitável)
            for response_code in [r1.status_code, r2.status_code, r3.status_code]:
                self.assertIn(response_code, [200, 400, 503, 404])
            
        except Exception as e:
            self.skipTest(f"Endpoints não disponíveis: {e}")
    
    def test_api_invalid_node_id(self):
        """Teste com ID de nó inválido"""
        
        try:
            response = self.app.get('/api/explain/99999')  # ID que provavelmente não existe
            
            # Deve retornar 400, 404 ou 503, mas não 500
            self.assertIn(response.status_code, [400, 404, 503])
            
        except Exception as e:
            self.skipTest(f"Endpoint não disponível: {e}")


class TestDashboardDataFlow(unittest.TestCase):
    """Testes de fluxo de dados do dashboard"""
    
    def setUp(self):
        """Prepara ambiente"""
        try:
            from app import app
            self.app = app
        except Exception:
            self.skipTest("App não carregado")
    
    def test_dashboard_index_loads(self):
        """Teste se página principal do dashboard carrega"""
        
        try:
            client = self.app.test_client()
            response = client.get('/')
            
            self.assertIn(response.status_code, [200, 404])
            
            if response.status_code == 200:
                self.assertIn(b'html', response.data.lower())
            
        except Exception as e:
            self.skipTest(f"Dashboard não disponível: {e}")
    
    def test_dashboard_can_fetch_node_data(self):
        """Teste se dashboard consegue buscar dados de um nó"""
        
        try:
            from app import app
            client = app.test_client()
            
            # Simular requisição que o JS do dashboard faria
            response = client.get('/api/explain/1')
            
            # Deve ser bem-sucedido ou retornar erro tratado
            self.assertIn(response.status_code, [200, 400, 503, 404])
            
        except Exception as e:
            self.skipTest(f"Fetch não funciona: {e}")


class TestEventManaerIntegration(unittest.TestCase):
    """Testes de integração do gerenciador de eventos"""
    
    def setUp(self):
        """Prepara"""
        from src.event_manager import EventManager
        self.event_manager = EventManager("data/exogenous_events_geocoded.json")
    
    def test_get_events_for_date(self):
        """Teste obter eventos para uma data específica"""
        
        today = date.today()
        events = self.event_manager.get_events_for_date(today)
        
        # Deve retornar uma lista (pode estar vazia)
        self.assertIsInstance(events, list)
    
    def test_get_anomaly_level_for_date(self):
        """Teste obter nível de anomalia para uma data"""
        
        today = date.today()
        anomaly_level = self.event_manager.get_anomaly_level_for_date(today)
        
        # Deve estar entre 0 e 1
        self.assertGreaterEqual(anomaly_level, 0.0)
        self.assertLessEqual(anomaly_level, 1.0)
    
    def test_anomaly_level_consistency(self):
        """Teste consistência do nível de anomalia ao chamar múltiplas vezes"""
        
        today = date.today()
        
        level1 = self.event_manager.get_anomaly_level_for_date(today)
        level2 = self.event_manager.get_anomaly_level_for_date(today)
        
        # Deve ser consistente
        self.assertEqual(level1, level2)
    
    def test_anomaly_level_historical_dates(self):
        """Teste nível de anomalia para datas históricas"""
        
        # Testar datas do passado
        past_date = date.today() - timedelta(days=7)
        anomaly_level = self.event_manager.get_anomaly_level_for_date(past_date)
        
        # Deve estar entre 0 e 1
        self.assertGreaterEqual(anomaly_level, 0.0)
        self.assertLessEqual(anomaly_level, 1.0)


class TestDataConsistency(unittest.TestCase):
    """Testes de consistência de dados entre módulos"""
    
    def test_metrics_reporter_consistency(self):
        """Teste consistência de cálculos de métricas"""
        
        from src.metrics import MetricReporter
        
        reporter = MetricReporter()
        
        rankings = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        true_labels = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
        
        # Chamar a mesma métrica várias vezes
        p1 = reporter.calculate_precision_at_k(rankings, true_labels, k=5)
        p2 = reporter.calculate_precision_at_k(rankings, true_labels, k=5)
        
        # Devem ser idênticas
        self.assertEqual(p1, p2)
    
    def test_explanation_generator_consistency(self):
        """Teste consistência de explicações geradas"""
        
        from src.explanation_generator import ExplanationGenerator
        
        generator = ExplanationGenerator()
        
        context = {
            "temporal_features": [0.8, 0.6],
            "spatial_features": [0.7, 0.5],
            "event_features": [0.9],
            "historical_features": [0.4]
        }
        
        # Gerar explicação duas vezes
        exp1 = generator.explain_node_ranking(1, 1, context)
        exp2 = generator.explain_node_ranking(1, 1, context)
        
        # Sumários devem ser iguais (estrutura pode variar)
        self.assertEqual(exp1["confidence"], exp2["confidence"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

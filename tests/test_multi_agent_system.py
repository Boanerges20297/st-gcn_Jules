import unittest
import json
import os
import sys

# Garantir que a raiz do projeto esteja no path para as importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.local_llm_client import LocalLLMClient
from src.agent.multi_agent_system import GeneralManagerAgent

class TestMultiAgentSystem(unittest.TestCase):
    """
    Conjunto de testes unitários para a Fase 6.1 (Multi-Agent System).
    Valida a blindagem dos especialistas e a integridade matemática da calibração.
    """
    def setUp(self):
        # Inicializa o Gerente Geral (que encapsula e blinda os especialistas)
        self.manager = GeneralManagerAgent()
        
    def test_client_real_connection_failure(self):
        """
        Valida que o cliente local do Ollama levanta exceções reais e legítimas
        (ConnectionError) quando o serviço local está offline, sem mocks ou fallbacks fictícios.
        """
        client = LocalLLMClient(base_url="http://localhost:9999", timeout=2) # URL inválida de propósito
        
        # Testando calibração (deve levantar ConnectionError)
        with self.assertRaises(ConnectionError):
            client.generate("calibre os pesos para mim")
        
    def test_specialist_encapsulation_and_blindness(self):
        """
        UAT: Valida a blindagem dos especialistas. 
        Garante que eles não sejam expostos fora do módulo do sistema multi-agente.
        """
        import src.agent.multi_agent_system as mas
        
        # O Gerente Geral deve ser a única interface pública importável
        self.assertTrue(hasattr(mas, "GeneralManagerAgent"))
        
        # Os especialistas devem começar com sub-traço (_) indicando privacidade
        self.assertFalse(hasattr(mas, "CalibrationAgent"))
        self.assertFalse(hasattr(mas, "InteractionAgent"))
        self.assertFalse(hasattr(mas, "ComplexDataAnalystAgent"))

    def test_full_pipeline_calibration(self):
        """
        Valida a orquestração completa em background do Gerente Geral.
        """
        raw_stgcn_data = {
            "confidence_scores": [0.91, 0.88, 0.72],
            "timestamp": "2026-05-27T08:00:00"
        }
        user_profile = {
            "region": "RMF",
            "focus": "CVLI",
            "historical_alerts": 1
        }
        
        response = self.manager.process_and_calibrate(raw_stgcn_data, user_profile)
        
        self.assertEqual(response["status"], "success")
        self.assertIn("calibrated_weights", response)
        self.assertIn("explanations", response)
        self.assertIn("data_analysis", response)
        
        # Verifica se os novos campos de skill do especialista analítico de CVLI estão presentes
        analyst_out = response["data_analysis"]
        self.assertIn("anomalies_detected", analyst_out)
        self.assertIn("geographical_drift", analyst_out)
        self.assertIn("next_probable_cvli_hotspot", analyst_out)
        self.assertIn("technical_summary", analyst_out)
        
        # Verifica estrutura dos pesos calibrados
        weights = response["calibrated_weights"]
        self.assertTrue(0.0 <= weights["posture"] <= 1.0)
        self.assertTrue(0.0 <= weights["speed"] <= 1.0)
        self.assertTrue(0.0 <= weights["rom"] <= 1.0)

if __name__ == '__main__':
    unittest.main()

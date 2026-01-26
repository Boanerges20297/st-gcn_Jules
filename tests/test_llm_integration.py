import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

class TestLLMIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.process_exogenous_text')
    @patch('app.find_node_coordinates')
    def test_parse_endpoint_flow(self, mock_find_coords, mock_process_text):
        # 1. Mock LLM response
        mock_process_text.return_value = [
            {
                'natureza': 'ROUBO',
                'localizacao_completa': 'RUA TESTE, BAIRRO TESTE',
                'bairro': 'BAIRRO TESTE',
                'municipio': 'FORTALEZA',
                'resumo': 'Roubo a pessoa',
                'raw': 'RAW TEXT'
            }
        ]

        # 2. Mock Coordinate lookup
        # Simulate finding coordinates for the neighborhood
        mock_find_coords.side_effect = lambda x: (-3.75, -38.55, 'specific') if 'TESTE' in x else None

        # 3. Call Endpoint
        payload = {'text': 'RAW TEXT'}
        response = self.app.post('/api/exogenous/parse',
                                 data=json.dumps(payload),
                                 content_type='application/json')

        data = json.loads(response.data)

        # 4. Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['events_processed'], 1)
        self.assertEqual(data['points_found'], 1)

        point = data['points'][0]
        self.assertEqual(point['lat'], -3.75)
        self.assertEqual(point['lng'], -38.55)
        self.assertIn('ROUBO', point['description'])
        self.assertIn('Roubo a pessoa', point['description'])

if __name__ == '__main__':
    unittest.main()

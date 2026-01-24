import unittest
import json
import sys
import os

sys.path.append(os.getcwd())
from app import app

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_simulation_endpoint(self):
        # Coordinates near Fortaleza
        payload = {
            'points': [[-3.7319, -38.5267]]
        }
        response = self.app.post('/api/simulate',
                                 data=json.dumps(payload),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('meta', data)
        # Check if risk scores are returned
        self.assertTrue(len(data['data']) > 0)
        item = data['data'][0]
        self.assertIn('risk_score', item)

if __name__ == '__main__':
    unittest.main()

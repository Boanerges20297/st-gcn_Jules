import sys
import os
import json
import unittest

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

class TestRiskAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_risk_endpoint_fields(self):
        print("\nTesting /api/risk structure...")
        response = self.client.get('/api/risk')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn('meta', data)
        meta = data['meta']
        self.assertIn('window_cvli', meta)
        self.assertIn('window_cvp', meta)
        self.assertIn('last_date', meta)
        self.assertIn('start_cvli', meta)
        self.assertIn('start_cvp', meta)

        self.assertIn('data', data)
        results = data['data']
        self.assertTrue(len(results) > 0)

        first = results[0]
        self.assertIn('risk_score_cvli', first)
        self.assertIn('risk_score_cvp', first)
        self.assertIn('cvli_pred', first)
        self.assertIn('cvp_pred', first)

        print("Meta Info:", meta)
        print("Sample Data Node 0:", first)

        # Sensitivity Check roughly
        # We can't easily check sensitivity without mock data, but we can check if fields exist.

if __name__ == '__main__':
    unittest.main()

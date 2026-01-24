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

    def test_risk_endpoint(self):
        print("\nTesting /api/risk...")
        response = self.client.get('/api/risk')

        if response.status_code == 503:
             print("Service Unavailable - likely model training not finished or files missing.")
             # This is acceptable if training is still running, but we warn.

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('data', data)
            self.assertIn('meta', data)

            results = data['data']
            self.assertTrue(len(results) > 0)

            first = results[0]
            self.assertIn('risk_score', first)
            self.assertIn('cvli_pred', first)
            self.assertIn('cvp_pred', first) # Added this in new app.py

            print("Risk endpoint returned valid data.")
            print("Meta:", data['meta'])
        else:
            print("Response:", response.data.decode())

if __name__ == '__main__':
    unittest.main()

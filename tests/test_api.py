import unittest
import json
import sys
import os

sys.path.append(os.getcwd())
from app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_polygons_endpoint(self):
        response = self.app.get('/api/polygons')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('type', data)
        self.assertEqual(data['type'], 'FeatureCollection')

    def test_risk_endpoint(self):
        response = self.app.get('/api/risk')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(isinstance(data, list))
        if len(data) > 0:
            item = data[0]
            self.assertIn('risk_score', item)
            self.assertIn('cvli_pred', item)
            self.assertIn('faction', item)

if __name__ == '__main__':
    unittest.main()

import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import format_trend

class TestTrendFormat(unittest.TestCase):
    def test_start_activity(self):
        # Avg 0, Pred > 0.001
        res = format_trend(0.005, 0)
        self.assertEqual(res, "Início de atividade detectada (Surgimento)")

    def test_stability_zero(self):
        # Avg 0, Pred 0
        res = format_trend(0, 0)
        self.assertEqual(res, "Estabilidade (Baixa Atividade)")

    def test_growth(self):
        # Avg 1, Pred 1.2 (+20%)
        res = format_trend(1.2, 1.0)
        self.assertEqual(res, "Tendência de Crescimento (+20.0%)")

    def test_reduction(self):
        # Avg 1, Pred 0.8 (-20%)
        res = format_trend(0.8, 1.0)
        self.assertEqual(res, "Tendência de Redução (-20.0%)")

    def test_stability_nonzero(self):
        # Avg 1, Pred 1.05 (+5%) -> Threshold is 10% in code
        res = format_trend(1.05, 1.0)
        self.assertEqual(res, "Estabilidade (+5.0%)")

if __name__ == '__main__':
    unittest.main()

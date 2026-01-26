import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import format_trend

class TestTrendFormat(unittest.TestCase):
    def test_start_activity(self):
        # Avg 0, Pred > 0.001
        res = format_trend(0.005, 0)
        self.assertEqual(res, "Início de atividade criminal")

    def test_stability_zero(self):
        # Avg 0, Pred 0
        res = format_trend(0, 0)
        self.assertEqual(res, "Estabilidade (Baixo Risco)")

    def test_growth(self):
        # Avg 1, Pred 1.2 (+20%) -> > 15%
        res = format_trend(1.2, 1.0)
        self.assertEqual(res, "Tendência de agravamento recente")

    def test_reduction(self):
        # Avg 1, Pred 0.8 (-20%) -> < -15%
        res = format_trend(0.8, 1.0)
        self.assertEqual(res, "Tendência de redução da atividade")

    def test_stability_low_risk(self):
        # Avg 1, Pred 1.05 (+5%) -> Stable
        # Risk Score Low (e.g. 10)
        res = format_trend(1.05, 1.0, risk_score=10.0)
        self.assertEqual(res, "Estabilidade (Baixo Risco)")

    def test_stability_medium_risk(self):
        # Avg 1, Pred 1.05 (+5%) -> Stable
        # Risk Score Medium (e.g. 30)
        res = format_trend(1.05, 1.0, risk_score=30.0)
        self.assertEqual(res, "Manutenção do padrão de risco médio")

    def test_stability_high_risk(self):
        # Avg 1, Pred 1.05 (+5%) -> Stable
        # Risk Score High (e.g. 80)
        res = format_trend(1.05, 1.0, risk_score=80.0)
        self.assertEqual(res, "Valor histórico alto para o período")

    def test_stability_no_risk_arg(self):
        # Avg 1, Pred 1.05 (+5%) -> Stable
        # No risk score passed -> default behavior
        res = format_trend(1.05, 1.0)
        self.assertEqual(res, "Estabilidade (Baixo Risco)")

if __name__ == '__main__':
    unittest.main()

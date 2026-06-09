import os
import sys
import tempfile
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.calibration_memory import (  # noqa: E402
    CalibrationSemanticMemory,
    record_agent_decision,
    record_validation_outcome,
)


class TestCalibrationSemanticMemory(unittest.TestCase):
    def test_records_positive_and_negative_experiences_for_prompt_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_validation_outcome(
                tmp,
                {
                    "region": "rmf",
                    "metric": "p20",
                    "current_value": 0.30,
                    "old_params": {"norm_neural_weight": 0.35},
                    "new_params": {
                        "norm_neural_weight": 0.85,
                        "dynamic_window": 30,
                        "use_historical_fallback": True,
                    },
                    "semantic_review": {"anomalies_detected": True},
                },
                {
                    "old_value": 0.30,
                    "new_value": 0.46,
                    "improvement_pct": 53.3,
                    "status": "improved",
                },
            )
            record_validation_outcome(
                tmp,
                {
                    "region": "rmf",
                    "metric": "p10",
                    "current_value": 0.20,
                    "old_params": {"tension_factor": 0.8},
                    "new_params": {
                        "tension_factor": 3.0,
                        "tag_bias_direct": 5.0,
                    },
                    "semantic_review": {"geographical_drift": True},
                },
                {
                    "old_value": 0.20,
                    "new_value": 0.18,
                    "improvement_pct": -10.0,
                    "status": "degraded",
                },
            )

            context = CalibrationSemanticMemory(tmp).build_context(region="rmf")

            self.assertEqual(context["events_considered"], 2)
            self.assertEqual(context["positive_patterns"][0]["outcome"], "improved")
            self.assertEqual(context["negative_patterns"][0]["outcome"], "degraded")
            tags = {item["tag"]: item["score"] for item in context["semantic_tag_scores"]}
            self.assertGreater(tags["rede_neural_dominante"], 0)
            self.assertLess(tags["tensao_alta"], 0)

    def test_records_agent_decision_as_pending_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_agent_decision(
                tmp,
                {
                    "target_region": "fortaleza",
                    "calibrated_weights": {"posture": 0.4, "speed": 0.3, "rom": 0.3},
                    "data_analysis": {"anomalies_detected": True},
                    "should_intervene": True,
                },
            )

            context = CalibrationSemanticMemory(tmp).build_context(region="fortaleza")

            self.assertEqual(context["events_considered"], 1)
            self.assertEqual(context["recent_events"][0]["outcome"], "pending_validation")
            self.assertIn("anomalia_semantica", context["recent_events"][0]["semantic_tags"])


if __name__ == "__main__":
    unittest.main()

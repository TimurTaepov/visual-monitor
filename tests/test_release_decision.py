import unittest

from vlm_evals.registry.release_decision import decide_release


class ReleaseDecisionTests(unittest.TestCase):
    def test_rejects_low_schema_validity(self):
        decision = decide_release(
            {
                "accuracy": 0.9,
                "schema_valid_rate": 0.8,
                "hallucination_rate": 0.02,
                "overconfident_wrong_rate": 0.01,
            }
        )
        self.assertEqual(decision["status"], "rejected")
        self.assertIn("schema_valid_rate_below_95_percent", decision["reasons"])

    def test_approves_good_metrics(self):
        decision = decide_release(
            {
                "accuracy": 0.9,
                "schema_valid_rate": 0.98,
                "hallucination_rate": 0.02,
                "overconfident_wrong_rate": 0.01,
            }
        )
        self.assertEqual(decision["status"], "approved_for_canary")


if __name__ == "__main__":
    unittest.main()


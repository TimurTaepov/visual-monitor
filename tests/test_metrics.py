import unittest

from vlm_evals.eval.metrics import compute_metrics, score_prediction
from vlm_evals.tasks.schemas import EvalTask


class MetricsTests(unittest.TestCase):
    def test_scores_primary_boolean_label(self):
        task = EvalTask(
            task_id="t1",
            image_path="image.jpg",
            task_type="retail_shelf_check",
            prompt_template="retail_shelf_v1",
            expected_schema="retail_shelf_schema",
            labels={"missing_stock": True},
            metadata={},
        )
        score = score_prediction(task, {"missing_stock": True, "confidence": 0.9})
        self.assertTrue(score["scoreable"])
        self.assertTrue(score["correct"])

    def test_compute_metrics(self):
        tasks = [
            EvalTask(
                task_id="t1",
                image_path="image.jpg",
                task_type="safety_helmet_check",
                prompt_template="object_presence_v1",
                expected_schema="object_presence_schema",
                labels={"answer": True},
                metadata={},
            )
        ]
        predictions = [
            {
                "task_id": "t1",
                "parsed_output": {"answer": True, "confidence": 0.9},
                "valid_json": True,
                "schema_valid": True,
                "latency_ms": 10.0,
            }
        ]
        metrics = compute_metrics(tasks, predictions)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["valid_json_rate"], 1.0)
        self.assertEqual(metrics["p95_latency_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()


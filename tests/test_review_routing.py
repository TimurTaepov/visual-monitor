import unittest

from vlm_evals.review.routing import ReviewPolicy, route_for_review
from vlm_evals.tasks.schemas import EvalTask


class ReviewRoutingTests(unittest.TestCase):
    def test_routes_low_confidence(self):
        task = EvalTask(
            task_id="t1",
            image_path="image.jpg",
            task_type="safety_helmet_check",
            prompt_template="object_presence_v1",
            expected_schema="object_presence_schema",
            labels={"answer": True},
            metadata={},
        )
        result = route_for_review(
            task,
            {
                "valid_json": True,
                "schema_valid": True,
                "parsed_output": {
                    "answer": True,
                    "confidence": 0.4,
                    "evidence": "visible",
                    "requires_review": False,
                },
            },
            ReviewPolicy(confidence_threshold=0.7),
        )
        self.assertTrue(result["requires_human_review"])
        self.assertIn("low_confidence", result["review_reasons"])


if __name__ == "__main__":
    unittest.main()


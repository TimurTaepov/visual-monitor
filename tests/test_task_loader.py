import tempfile
import unittest
from pathlib import Path

from vlm_evals.tasks.loader import load_tasks, write_jsonl


class TaskLoaderTests(unittest.TestCase):
    def test_loads_jsonl_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "task_id": "t1",
                        "image_path": "image.jpg",
                        "task_type": "retail_shelf_check",
                        "prompt_template": "retail_shelf_v1",
                        "expected_schema": "retail_shelf_schema",
                        "labels": {"missing_stock": True},
                        "metadata": {},
                    }
                ],
            )
            tasks = load_tasks(path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task_id, "t1")


if __name__ == "__main__":
    unittest.main()


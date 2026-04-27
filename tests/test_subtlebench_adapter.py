import json
import tempfile
import unittest
from pathlib import Path

from vlm_evals.datasets.subtlebench import load_subtlebench_tasks


class SubtleBenchAdapterTests(unittest.TestCase):
    def test_loads_two_image_multiple_choice_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            image_dir = root / "images"
            data_dir.mkdir()
            image_dir.mkdir()
            (image_dir / "a.jpg").write_bytes(b"a")
            (image_dir / "b.jpg").write_bytes(b"b")
            row = {
                "image_1": "images/a.jpg",
                "image_2": "images/b.jpg",
                "question": "In which image is the box open?",
                "answer": "second image",
                "distractors": ["first image"],
                "has_caption": True,
                "caption": "The box is open in the second image.",
                "category": "state",
                "domain": "natural",
                "source": "unit",
                "source_id": "1",
            }
            with (data_dir / "test.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            tasks = load_subtlebench_tasks({"path": str(root), "split": "test"})

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_type, "subtlebench_multiple_choice")
        self.assertEqual(task.expected_schema, "subtlebench_multiple_choice_schema")
        self.assertEqual(task.labels["answer"], "second image")
        self.assertEqual(task.metadata["choices"], ["first image", "second image"])
        self.assertEqual(len(task.image_paths), 2)

    def test_skips_missing_local_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            row = {
                "image_1": "images/missing-a.jpg",
                "image_2": "images/missing-b.jpg",
                "question": "In which image is the box open?",
                "answer": "first image",
                "distractors": ["second image"],
            }
            with (data_dir / "test.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            tasks = load_subtlebench_tasks({"path": str(root), "split": "test"})

        self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()

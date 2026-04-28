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

    def test_filters_image_choice_answers_and_balances_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            image_dir = root / "images"
            data_dir.mkdir()
            image_dir.mkdir()
            (image_dir / "a.jpg").write_bytes(b"a")
            (image_dir / "b.jpg").write_bytes(b"b")
            rows = [
                {
                    "image_1": "images/a.jpg",
                    "image_2": "images/b.jpg",
                    "question": "q1",
                    "answer": "The first image.",
                    "category": "state",
                    "source": "unit",
                    "source_id": "1",
                },
                {
                    "image_1": "images/a.jpg",
                    "image_2": "images/b.jpg",
                    "question": "q2",
                    "answer": "second image",
                    "category": "quantity",
                    "source": "unit",
                    "source_id": "2",
                },
                {
                    "image_1": "images/a.jpg",
                    "image_2": "images/b.jpg",
                    "question": "q3",
                    "answer": "A free-form caption answer.",
                    "category": "state",
                    "source": "unit",
                    "source_id": "3",
                },
                {
                    "image_1": "images/a.jpg",
                    "image_2": "images/b.jpg",
                    "question": "q4",
                    "answer": "the second image",
                    "category": "state",
                    "source": "unit",
                    "source_id": "1",
                },
            ]
            with (data_dir / "test.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            tasks = load_subtlebench_tasks(
                {
                    "path": str(root),
                    "split": "test",
                    "answer_mode": "image_choice",
                    "sample_strategy": "balanced_by_category",
                    "max_examples": 3,
                }
            )

        self.assertEqual(len(tasks), 3)
        self.assertEqual(len({task.task_id for task in tasks}), 3)
        self.assertEqual([task.metadata["category"] for task in tasks], ["quantity", "state", "state"])
        self.assertEqual([task.labels["answer"] for task in tasks], ["second image", "first image", "second image"])


if __name__ == "__main__":
    unittest.main()

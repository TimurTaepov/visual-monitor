from __future__ import annotations

from typing import Any

from vlm_evals.datasets.subtlebench import load_subtlebench_tasks
from vlm_evals.tasks.loader import load_tasks
from vlm_evals.tasks.schemas import EvalTask


def load_eval_tasks(config: dict[str, Any]) -> list[EvalTask]:
    dataset_config = config.get("dataset")
    if not dataset_config:
        return load_tasks(config["tasks_path"], config.get("task_types"))

    dataset_name = str(dataset_config.get("name", "")).lower()
    if dataset_name == "subtlebench":
        return load_subtlebench_tasks(dataset_config)
    raise ValueError(f"Unknown dataset name {dataset_name!r}")

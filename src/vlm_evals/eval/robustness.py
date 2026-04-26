from __future__ import annotations

from collections import defaultdict
from typing import Any

from vlm_evals.eval.metrics import compute_metrics
from vlm_evals.tasks.schemas import EvalTask


def metrics_by_metadata(
    tasks: list[EvalTask],
    predictions: list[dict[str, Any]],
    metadata_key: str,
) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    grouped_tasks: dict[str, list[EvalTask]] = defaultdict(list)
    grouped_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for task in tasks:
        value = str(task.metadata.get(metadata_key, "unknown"))
        grouped_tasks[value].append(task)
    for prediction in predictions:
        task = task_by_id.get(str(prediction.get("task_id")))
        if task:
            value = str(task.metadata.get(metadata_key, "unknown"))
            grouped_predictions[value].append(prediction)
    return {
        value: compute_metrics(grouped_tasks[value], grouped_predictions.get(value, []))
        for value in sorted(grouped_tasks)
    }


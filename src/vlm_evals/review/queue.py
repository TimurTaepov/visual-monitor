from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_evals.tasks.loader import write_jsonl


def export_review_queue(predictions: list[dict[str, Any]], path: str | Path) -> list[dict[str, Any]]:
    queue = [prediction for prediction in predictions if prediction.get("requires_human_review")]
    write_jsonl(path, queue)
    return queue


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from vlm_evals.tasks.schemas import EvalTask


def load_tasks(path: str | Path, task_types: Iterable[str] | None = None) -> list[EvalTask]:
    path = Path(path)
    allowed = set(task_types or [])
    tasks: list[EvalTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                task = EvalTask.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid task JSONL at {path}:{line_number}: {exc}") from exc
            if allowed and task.task_type not in allowed:
                continue
            tasks.append(task)
    return tasks


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


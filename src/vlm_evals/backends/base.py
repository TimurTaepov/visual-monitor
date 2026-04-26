from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vlm_evals.tasks.schemas import EvalTask


class VisionModelBackend(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.backend_name = str(config.get("backend", self.__class__.__name__.lower()))
        self.model_name = str(config.get("model_name", "unknown"))

    @abstractmethod
    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "config": {k: v for k, v in self.config.items() if "key" not in k.lower()},
        }


def prediction_record(
    *,
    task: EvalTask,
    backend: str,
    model_name: str,
    raw_output: str,
    latency_ms: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    estimated_cost_usd: float = 0.0,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "backend": backend,
        "model_name": model_name,
        "prompt_id": task.prompt_template,
        "task_type": task.task_type,
        "expected_schema": task.expected_schema,
        "raw_output": raw_output,
        "parsed_output": None,
        "valid_json": False,
        "schema_valid": False,
        "latency_ms": round(latency_ms, 3),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "estimated_cost_usd": estimated_cost_usd,
        "error": error,
    }


from __future__ import annotations

import hashlib
import json
from typing import Any

from vlm_evals.backends.base import VisionModelBackend, prediction_record
from vlm_evals.tasks.schemas import EvalTask
from vlm_evals.utils.cost import estimate_request_cost
from vlm_evals.utils.timing import Timer


class MockBackend(VisionModelBackend):
    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        with Timer() as timer:
            parsed = self._make_output(task)
            raw_output = json.dumps(parsed)

        return prediction_record(
            task=task,
            backend="mock",
            model_name=self.model_name,
            raw_output=raw_output,
            latency_ms=timer.elapsed_ms,
            tokens_in=max(1, len(prompt.split())),
            tokens_out=max(1, len(raw_output.split())),
            estimated_cost_usd=estimate_request_cost(self.config),
        )

    def _make_output(self, task: EvalTask) -> dict[str, Any]:
        behavior = self.config.get("behavior", "label_oracle")
        flip = behavior == "noisy" and self._should_flip(task.task_id)
        confidence = 0.93 if not flip else 0.58
        requires_review = confidence < 0.7 or bool(task.metadata.get("is_hard_case", False))

        if task.expected_schema.startswith("retail_shelf"):
            missing_stock = bool(task.labels.get("missing_stock", False))
            if flip:
                missing_stock = not missing_stock
            return {
                "missing_stock": missing_stock,
                "confidence": confidence,
                "evidence": self._evidence(task, "retail shelf state"),
                "requires_review": requires_review,
            }

        if task.expected_schema.startswith("safety_check"):
            anomaly_present = bool(task.labels.get("anomaly_present", False))
            if flip:
                anomaly_present = not anomaly_present
            risk_level = task.labels.get("risk_level", "medium" if anomaly_present else "low")
            if flip:
                risk_level = "medium" if risk_level == "low" else "low"
            return {
                "risk_level": risk_level,
                "anomaly_present": anomaly_present,
                "confidence": confidence,
                "evidence": self._evidence(task, "industrial visual condition"),
                "requires_review": requires_review,
            }

        answer = bool(task.labels.get("answer", False))
        if flip:
            answer = not answer
        return {
            "answer": answer,
            "confidence": confidence,
            "evidence": self._evidence(task, "visible object condition"),
            "requires_review": requires_review,
        }

    def _should_flip(self, task_id: str) -> bool:
        error_rate = float(self.config.get("error_rate", 0.2) or 0.0)
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return bucket < error_rate

    @staticmethod
    def _evidence(task: EvalTask, noun: str) -> str:
        source = task.metadata.get("source_dataset", "sample")
        return f"Mock evidence derived from {source} labels for {noun}."


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_evals.tasks.schemas import EvalTask


@dataclass
class ReviewPolicy:
    confidence_threshold: float = 0.7
    high_risk_task_types: set[str] = field(default_factory=set)


def route_for_review(
    task: EvalTask,
    prediction: dict[str, Any],
    policy: ReviewPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ReviewPolicy()
    reasons: list[str] = []
    parsed = prediction.get("parsed_output")

    if prediction.get("error"):
        reasons.append("backend_error")
    if not prediction.get("valid_json"):
        reasons.append("invalid_json")
    if not prediction.get("schema_valid"):
        reasons.append("schema_validation_failed")

    if isinstance(parsed, dict):
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        if confidence < policy.confidence_threshold:
            reasons.append("low_confidence")
        if parsed.get("requires_review") is True:
            reasons.append("model_requested_review")
        if task.task_type in policy.high_risk_task_types and parsed.get("anomaly_present") is True:
            reasons.append("high_risk_positive")
    elif not prediction.get("error"):
        reasons.append("missing_parsed_output")

    return {
        "requires_human_review": bool(reasons),
        "review_reasons": sorted(set(reasons)),
    }


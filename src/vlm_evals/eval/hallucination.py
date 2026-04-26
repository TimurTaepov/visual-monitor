from __future__ import annotations

from typing import Any

from vlm_evals.eval.metrics import score_prediction
from vlm_evals.tasks.schemas import EvalTask


def hallucination_flags(task: EvalTask, prediction: dict[str, Any]) -> dict[str, Any]:
    parsed = prediction.get("parsed_output")
    score = score_prediction(task, parsed)
    confidence = 0.0
    evidence = ""
    if isinstance(parsed, dict):
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        evidence = str(parsed.get("evidence", "") or "")

    wrong = score.get("scoreable") and score.get("correct") is False
    overconfident_wrong = bool(wrong and confidence >= 0.8)
    unsupported_evidence = bool(wrong and evidence.strip() and confidence >= 0.7)
    hallucination = overconfident_wrong or unsupported_evidence
    return {
        "hallucination": hallucination,
        "unsupported_evidence": unsupported_evidence,
        "overconfident_wrong": overconfident_wrong,
    }


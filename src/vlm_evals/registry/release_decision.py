from __future__ import annotations

from typing import Any


def decide_release(metrics: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    reasons: list[str] = []
    status = "approved_for_canary"

    accuracy = metrics.get("accuracy")
    schema_valid_rate = metrics.get("schema_valid_rate")
    hallucination_rate = metrics.get("hallucination_rate")
    overconfident_wrong_rate = metrics.get("overconfident_wrong_rate")
    review_rate = metrics.get("review_rate")
    p95_latency_ms = metrics.get("p95_latency_ms")

    if accuracy is None or accuracy < 0.8:
        status = "needs_more_data" if accuracy is None else "rejected"
        reasons.append("accuracy_below_threshold")
    if schema_valid_rate is None or schema_valid_rate < 0.95:
        status = "rejected"
        reasons.append("schema_valid_rate_below_95_percent")
    if hallucination_rate is not None and hallucination_rate > 0.10:
        status = "rejected"
        reasons.append("hallucination_rate_above_10_percent")
    if overconfident_wrong_rate is not None and overconfident_wrong_rate > 0.07:
        status = "rejected"
        reasons.append("overconfident_wrong_rate_above_7_percent")

    if baseline:
        baseline_accuracy = baseline.get("accuracy")
        baseline_p95 = baseline.get("p95_latency_ms")
        baseline_review_rate = baseline.get("review_rate")
        if baseline_accuracy is not None and accuracy is not None:
            if accuracy < baseline_accuracy + 0.03:
                status = "needs_more_data" if status == "approved_for_canary" else status
                reasons.append("accuracy_does_not_improve_baseline_by_3_percent")
        if baseline_p95 and p95_latency_ms and p95_latency_ms > baseline_p95 * 1.2:
            status = "needs_more_data" if status == "approved_for_canary" else status
            reasons.append("p95_latency_regression_above_20_percent")
        if baseline_review_rate is not None and review_rate is not None:
            if review_rate > baseline_review_rate + 0.10:
                status = "needs_more_data" if status == "approved_for_canary" else status
                reasons.append("review_rate_regression_above_10_percent")

    if not reasons:
        reasons.append("all_canary_thresholds_met")
    return {"status": status, "reasons": reasons}


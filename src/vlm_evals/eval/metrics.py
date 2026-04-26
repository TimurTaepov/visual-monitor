from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from vlm_evals.tasks.schemas import EvalTask

TASK_PRIMARY_LABEL: dict[str, str] = {
    "safety_helmet_check": "answer",
    "retail_shelf_check": "missing_stock",
    "industrial_safety_check": "anomaly_present",
}


def primary_label_field(task: EvalTask) -> str | None:
    return TASK_PRIMARY_LABEL.get(task.task_type)


def score_prediction(task: EvalTask, parsed_output: dict[str, Any] | None) -> dict[str, Any]:
    field = primary_label_field(task)
    if field is None or parsed_output is None:
        return {"scoreable": False, "correct": None, "label_field": field}
    expected = task.labels.get(field)
    actual = parsed_output.get(field)
    if expected is None or actual is None:
        return {"scoreable": False, "correct": None, "label_field": field}
    return {
        "scoreable": True,
        "correct": actual == expected,
        "label_field": field,
        "expected": expected,
        "actual": actual,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 3)


def _binary_metrics(scored: list[dict[str, Any]]) -> dict[str, float | None]:
    bool_items = [
        item
        for item in scored
        if isinstance(item.get("expected"), bool) and isinstance(item.get("actual"), bool)
    ]
    if not bool_items:
        return {"precision": None, "recall": None, "f1": None}

    tp = sum(1 for item in bool_items if item["expected"] is True and item["actual"] is True)
    fp = sum(1 for item in bool_items if item["expected"] is False and item["actual"] is True)
    fn = sum(1 for item in bool_items if item["expected"] is True and item["actual"] is False)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_metrics(tasks: list[EvalTask], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    scored: list[dict[str, Any]] = []
    latencies = [float(p.get("latency_ms", 0.0) or 0.0) for p in predictions if p.get("error") is None]

    for prediction in predictions:
        task = task_by_id.get(str(prediction.get("task_id")))
        if not task:
            continue
        score = score_prediction(task, prediction.get("parsed_output"))
        prediction["score"] = score
        if score.get("scoreable"):
            scored.append(score)

    correct = sum(1 for item in scored if item.get("correct") is True)
    accuracy = correct / len(scored) if scored else None
    valid_json_count = sum(1 for p in predictions if p.get("valid_json") is True)
    schema_valid_count = sum(1 for p in predictions if p.get("schema_valid") is True)
    review_count = sum(1 for p in predictions if p.get("requires_human_review") is True)
    hallucination_count = sum(1 for p in predictions if p.get("hallucination") is True)
    overconfident_wrong_count = sum(1 for p in predictions if p.get("overconfident_wrong") is True)
    error_count = sum(1 for p in predictions if p.get("error"))

    binary = _binary_metrics(scored)
    total = len(predictions)
    metrics: dict[str, Any] = {
        "num_examples": len(tasks),
        "num_predictions": total,
        "scoreable_examples": len(scored),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "valid_json_rate": round(valid_json_count / total, 4) if total else None,
        "schema_valid_rate": round(schema_valid_count / total, 4) if total else None,
        "review_rate": round(review_count / total, 4) if total else None,
        "hallucination_rate": round(hallucination_count / total, 4) if total else None,
        "overconfident_wrong_rate": round(overconfident_wrong_count / total, 4) if total else None,
        "error_rate": round(error_count / total, 4) if total else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "avg_latency_ms": round(mean(latencies), 3) if latencies else None,
        "estimated_cost_usd": round(
            sum(float(p.get("estimated_cost_usd", 0.0) or 0.0) for p in predictions), 6
        ),
    }
    metrics.update(binary)
    return metrics


def metrics_by_task_type(tasks: list[EvalTask], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    task_groups: dict[str, list[EvalTask]] = defaultdict(list)
    pred_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        task_groups[task.task_type].append(task)
    for prediction in predictions:
        task = task_by_id.get(str(prediction.get("task_id")))
        if task:
            pred_groups[task.task_type].append(prediction)
    return {
        task_type: compute_metrics(task_groups[task_type], pred_groups.get(task_type, []))
        for task_type in sorted(task_groups)
    }


def confusion_counts(tasks: list[EvalTask], predictions: list[dict[str, Any]]) -> dict[str, int]:
    task_by_id = {task.task_id: task for task in tasks}
    counts: Counter[str] = Counter()
    for prediction in predictions:
        task = task_by_id.get(str(prediction.get("task_id")))
        if not task:
            continue
        score = score_prediction(task, prediction.get("parsed_output"))
        if not score.get("scoreable") or not isinstance(score.get("expected"), bool):
            continue
        expected = bool(score["expected"])
        actual = bool(score["actual"])
        if expected and actual:
            counts["tp"] += 1
        elif not expected and actual:
            counts["fp"] += 1
        elif expected and not actual:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return dict(counts)


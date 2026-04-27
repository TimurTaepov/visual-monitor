from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_evals.eval.metrics import ScoringConfig, confusion_counts, metrics_by_task_type
from vlm_evals.eval.robustness import metrics_by_metadata
from vlm_evals.tasks.schemas import EvalTask


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}" if value <= 1 else f"{value:.1f}"
    return str(value)


def build_report(
    *,
    run_id: str,
    backend_info: dict[str, Any],
    tasks: list[EvalTask],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    scoring: ScoringConfig | None = None,
) -> str:
    by_task = metrics_by_task_type(tasks, predictions, scoring)
    by_category = metrics_by_metadata(tasks, predictions, "category", scoring)
    by_domain = metrics_by_metadata(tasks, predictions, "domain", scoring)
    failures = [
        p
        for p in predictions
        if p.get("score", {}).get("scoreable") and p.get("score", {}).get("correct") is False
    ][:10]
    confusion = confusion_counts(tasks, predictions, scoring)

    lines: list[str] = [
        f"# VLM Evaluation Report: {run_id}",
        "",
        "## Backend",
        "",
        f"- Backend: `{backend_info.get('backend')}`",
        f"- Model: `{backend_info.get('model_name')}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "num_examples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "valid_json_rate",
        "schema_valid_rate",
        "hallucination_rate",
        "overconfident_wrong_rate",
        "review_rate",
        "error_rate",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "estimated_cost_usd",
    ]:
        lines.append(f"| {key} | {format_metric(metrics.get(key))} |")

    lines.extend(["", "## Metrics By Task Type", "", "| Task Type | Accuracy | F1 | JSON Valid | Review Rate | p95 Latency |", "|---|---:|---:|---:|---:|---:|"])
    for task_type, task_metrics in by_task.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    task_type,
                    format_metric(task_metrics.get("accuracy")),
                    format_metric(task_metrics.get("f1")),
                    format_metric(task_metrics.get("valid_json_rate")),
                    format_metric(task_metrics.get("review_rate")),
                    format_metric(task_metrics.get("p95_latency_ms")),
                ]
            )
            + " |"
        )

    if any(value != "unknown" for value in by_category):
        lines.extend(["", "## Metrics By Category", "", "| Category | Accuracy | JSON Valid | p95 Latency |", "|---|---:|---:|---:|"])
        for category, category_metrics in by_category.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        category,
                        format_metric(category_metrics.get("accuracy")),
                        format_metric(category_metrics.get("valid_json_rate")),
                        format_metric(category_metrics.get("p95_latency_ms")),
                    ]
                )
                + " |"
            )

    if any(value != "unknown" for value in by_domain):
        lines.extend(["", "## Metrics By Domain", "", "| Domain | Accuracy | JSON Valid | p95 Latency |", "|---|---:|---:|---:|"])
        for domain, domain_metrics in by_domain.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        domain,
                        format_metric(domain_metrics.get("accuracy")),
                        format_metric(domain_metrics.get("valid_json_rate")),
                        format_metric(domain_metrics.get("p95_latency_ms")),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Confusion Counts", ""])
    if confusion:
        lines.extend(f"- {key}: {value}" for key, value in sorted(confusion.items()))
    else:
        lines.append("- n/a")

    lines.extend(["", "## Example Failures", ""])
    if not failures:
        lines.append("No scored failures in this run.")
    else:
        for failure in failures:
            score = failure.get("score", {})
            lines.extend(
                [
                    f"### {failure.get('task_id')}",
                    "",
                    f"- Expected: `{score.get('expected')}`",
                    f"- Actual: `{score.get('actual')}`",
                    f"- Review reasons: `{', '.join(failure.get('review_reasons', []))}`",
                    f"- Raw output: `{failure.get('raw_output')}`",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Deployment Recommendation",
            "",
            deployment_recommendation(metrics),
            "",
        ]
    )
    return "\n".join(lines)


def deployment_recommendation(metrics: dict[str, Any]) -> str:
    accuracy = metrics.get("accuracy") or 0.0
    schema_rate = metrics.get("schema_valid_rate") or 0.0
    hallucination_rate = metrics.get("hallucination_rate") or 0.0
    if accuracy >= 0.85 and schema_rate >= 0.95 and hallucination_rate <= 0.10:
        return "Approve for canary on low-risk tasks with human review enabled."
    if schema_rate < 0.90:
        return "Reject for canary until structured-output reliability improves."
    return "Needs more data or prompt/backend tuning before canary."


def write_report(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

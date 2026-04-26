from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vlm_evals.backends import create_backend, load_backend_config
from vlm_evals.backends.base import prediction_record
from vlm_evals.eval.hallucination import hallucination_flags
from vlm_evals.eval.json_validation import parse_and_validate
from vlm_evals.eval.metrics import compute_metrics, score_prediction
from vlm_evals.eval.report import build_report, write_report
from vlm_evals.review.routing import ReviewPolicy, route_for_review
from vlm_evals.tasks.loader import load_tasks, write_jsonl
from vlm_evals.tasks.prompt_builder import PromptBuilder
from vlm_evals.tasks.schemas import EvalTask, schema_json_for
from vlm_evals.utils.config import ensure_dir, load_yaml
from vlm_evals.utils.timing import Timer


def _review_policy(config: dict[str, Any]) -> ReviewPolicy:
    review = config.get("review", {}) or {}
    return ReviewPolicy(
        confidence_threshold=float(review.get("confidence_threshold", 0.7)),
        high_risk_task_types=set(review.get("high_risk_task_types", [])),
    )


def _error_prediction(task: EvalTask, backend_info: dict[str, Any], error: Exception, elapsed_ms: float) -> dict[str, Any]:
    return prediction_record(
        task=task,
        backend=str(backend_info.get("backend", "unknown")),
        model_name=str(backend_info.get("model_name", "unknown")),
        raw_output="",
        latency_ms=elapsed_ms,
        error=f"{error.__class__.__name__}: {error}",
    )


def _finalize_prediction(
    *,
    task: EvalTask,
    prediction: dict[str, Any],
    policy: ReviewPolicy,
) -> dict[str, Any]:
    validation = parse_and_validate(prediction.get("raw_output"), task.expected_schema)
    prediction.update(validation)
    prediction.update(hallucination_flags(task, prediction))
    prediction.update(route_for_review(task, prediction, policy))
    prediction["score"] = score_prediction(task, prediction.get("parsed_output"))
    return prediction


def evaluate_backend(
    *,
    config: dict[str, Any],
    backend_path: str | Path | dict[str, Any],
    tasks: list[EvalTask],
) -> dict[str, Any]:
    backend_config = load_backend_config(backend_path)
    backend = create_backend(backend_config)
    backend_info = backend.model_info()
    prompt_builder = PromptBuilder(
        prompt_dir=config.get("prompt_dir", "prompts"),
        default_prompt_id=config.get("default_prompt_id"),
    )
    policy = _review_policy(config)
    predictions: list[dict[str, Any]] = []

    try:
        backend.start()
    except Exception as exc:
        try:
            backend.close()
        except Exception:
            pass
        for task in tasks:
            prediction = _error_prediction(task, backend_info, exc, 0.0)
            predictions.append(_finalize_prediction(task=task, prediction=prediction, policy=policy))
        metrics = compute_metrics(tasks, predictions)
        return {
            "backend_info": backend_info,
            "backend_config": backend_config,
            "predictions": predictions,
            "metrics": metrics,
        }

    close_error: str | None = None
    try:
        for task in tasks:
            prompt = prompt_builder.build(task)
            schema = schema_json_for(task.expected_schema)
            timer = Timer()
            try:
                with timer:
                    prediction = backend.predict(task, prompt, schema)
            except Exception as exc:
                prediction = _error_prediction(task, backend_info, exc, timer.elapsed_ms)

            predictions.append(_finalize_prediction(task=task, prediction=prediction, policy=policy))
    finally:
        try:
            backend.close()
        except Exception as exc:
            close_error = f"{exc.__class__.__name__}: {exc}"

    if close_error:
        for prediction in predictions:
            prediction["backend_close_error"] = close_error

    metrics = compute_metrics(tasks, predictions)
    return {
        "backend_info": backend_info,
        "backend_config": backend_config,
        "predictions": predictions,
        "metrics": metrics,
    }


def run_eval(config_path: str | Path = "configs/eval/default.yaml") -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    tasks = load_tasks(config["tasks_path"], config.get("task_types"))
    output_dir = ensure_dir(config.get("output_dir", "reports"))
    backend_paths = config.get("backends", [])
    if not backend_paths:
        raise ValueError("Evaluation config must include at least one backend")

    all_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    for backend_path in backend_paths:
        result = evaluate_backend(config=config, backend_path=backend_path, tasks=tasks)
        backend_info = result["backend_info"]
        safe_model = str(backend_info["model_name"]).replace("/", "_").replace(":", "_")
        predictions_path = Path(
            config.get("predictions_path") or output_dir / f"{safe_model}_predictions.jsonl"
        )
        report_path = Path(config.get("report_path") or output_dir / f"{safe_model}_report.md")
        if len(backend_paths) > 1 and config.get("predictions_path"):
            predictions_path = output_dir / f"{safe_model}_predictions.jsonl"
        if len(backend_paths) > 1 and config.get("report_path"):
            report_path = output_dir / f"{safe_model}_report.md"

        write_jsonl(predictions_path, result["predictions"])
        report = build_report(
            run_id=str(config.get("run_id", config_path.stem)),
            backend_info=backend_info,
            tasks=tasks,
            predictions=result["predictions"],
            metrics=result["metrics"],
        )
        write_report(report_path, report)
        result["predictions_path"] = str(predictions_path)
        result["report_path"] = str(report_path)
        all_predictions.extend(result["predictions"])
        all_results.append(result)

    summary = {
        "config_path": str(config_path),
        "num_tasks": len(tasks),
        "results": all_results,
        "combined_metrics": compute_metrics(tasks * len(all_results), all_predictions)
        if all_results
        else {},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/default.yaml")
    args = parser.parse_args()
    summary = run_eval(args.config)
    for result in summary["results"]:
        backend = result["backend_info"]
        metrics = result["metrics"]
        print(
            f"{backend['backend']}:{backend['model_name']} "
            f"accuracy={metrics.get('accuracy')} "
            f"json={metrics.get('valid_json_rate')} "
            f"report={result['report_path']}"
        )


if __name__ == "__main__":
    main()

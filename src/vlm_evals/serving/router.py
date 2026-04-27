from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from vlm_evals.backends import create_backend
from vlm_evals.eval.hallucination import hallucination_flags
from vlm_evals.eval.json_validation import parse_and_validate
from vlm_evals.eval.metrics import score_prediction
from vlm_evals.eval.run_eval import run_eval
from vlm_evals.review.routing import ReviewPolicy, route_for_review
from vlm_evals.serving.schemas import BatchPredictRequest, PredictRequest
from vlm_evals.tasks.loader import load_tasks
from vlm_evals.tasks.prompt_builder import PromptBuilder
from vlm_evals.tasks.schemas import EvalTask, schema_json_for

router = APIRouter()

TASK_DEFAULTS = {
    "safety_helmet_check": ("object_presence_v1", "object_presence_schema"),
    "retail_shelf_check": ("retail_shelf_v1", "retail_shelf_schema"),
    "industrial_safety_check": ("safety_check_v1", "safety_check_schema"),
}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models")
def models() -> dict[str, list[str]]:
    configs = sorted(str(path) for path in Path("configs/backends").glob("*.yaml"))
    return {"backend_configs": configs}


@router.get("/tasks")
def tasks() -> dict[str, list[dict[str, str]]]:
    loaded = load_tasks("data/tasks/eval_tasks.jsonl")
    return {
        "tasks": [
            {"task_id": task.task_id, "task_type": task.task_type, "image_paths": task.image_paths}
            for task in loaded
        ]
    }


def _make_task(request: PredictRequest) -> EvalTask:
    default_prompt, default_schema = TASK_DEFAULTS.get(
        request.task_type, ("object_presence_v1", "object_presence_schema")
    )
    return EvalTask(
        task_id="api_request",
        image_path=request.image_path,
        image_paths=request.image_paths,
        task_type=request.task_type,
        prompt_template=request.prompt_id or default_prompt,
        expected_schema=request.expected_schema or default_schema,
        labels=request.labels,
        metadata=request.metadata,
    )


def predict_one(request: PredictRequest) -> dict:
    task = _make_task(request)
    backend = create_backend(request.backend_config)
    try:
        backend.start()
        prompt = PromptBuilder().build(task)
        prediction = backend.predict(task, prompt, schema_json_for(task.expected_schema))
        prediction.update(parse_and_validate(prediction.get("raw_output"), task.expected_schema))
        prediction.update(hallucination_flags(task, prediction))
        prediction.update(route_for_review(task, prediction, ReviewPolicy()))
        prediction["score"] = score_prediction(task, prediction.get("parsed_output"))
        return prediction
    finally:
        backend.close()


@router.post("/predict")
def predict(request: PredictRequest) -> dict:
    return predict_one(request)


@router.post("/batch_predict")
def batch_predict(request: BatchPredictRequest) -> dict:
    return {"predictions": [predict_one(item) for item in request.requests]}


@router.post("/evaluate")
def evaluate() -> dict:
    summary = run_eval("configs/eval/default.yaml")
    return {
        "num_tasks": summary["num_tasks"],
        "results": [
            {
                "backend": result["backend_info"]["backend"],
                "model_name": result["backend_info"]["model_name"],
                "metrics": result["metrics"],
                "report_path": result["report_path"],
            }
            for result in summary["results"]
        ],
    }


@router.get("/metrics")
def metrics() -> dict:
    return {"status": "ok", "note": "Service counters can be wired here for deployment."}

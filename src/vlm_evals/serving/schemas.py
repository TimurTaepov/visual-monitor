from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_path: str = ""
    image_paths: list[str] = Field(default_factory=list)
    task_type: str
    backend_config: str
    prompt_id: str | None = None
    expected_schema: str | None = None
    labels: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest]

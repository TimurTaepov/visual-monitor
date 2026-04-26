from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_path: str
    task_type: str
    backend_config: str = "configs/backends/mock_oracle.yaml"
    prompt_id: str | None = None
    expected_schema: str | None = None
    labels: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest]


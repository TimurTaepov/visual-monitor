from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class EvalTask(BaseModel):
    task_id: str
    image_path: str = ""
    image_paths: list[str] = Field(default_factory=list)
    task_type: str
    prompt_template: str
    expected_schema: str
    labels: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_image_paths(self) -> "EvalTask":
        if not self.image_paths and self.image_path:
            self.image_paths = [self.image_path]
        elif self.image_paths and not self.image_path:
            self.image_path = self.image_paths[0]
        if not self.image_paths:
            raise ValueError("EvalTask must include image_path or image_paths")
        return self


class ObjectPresenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    requires_review: bool


class RetailShelfOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    missing_stock: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    requires_review: bool


class SafetyCheckOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: Literal["low", "medium", "high"]
    anomaly_present: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    requires_review: bool


class MultipleChoiceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    requires_review: bool


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "object_presence_schema": ObjectPresenceOutput,
    "object_presence_schema_v1": ObjectPresenceOutput,
    "retail_shelf_schema": RetailShelfOutput,
    "retail_shelf_schema_v1": RetailShelfOutput,
    "safety_check_schema": SafetyCheckOutput,
    "safety_check_schema_v1": SafetyCheckOutput,
    "multiple_choice_schema": MultipleChoiceOutput,
    "subtlebench_multiple_choice_schema": MultipleChoiceOutput,
}


def normalize_schema_id(schema_id: str) -> str:
    schema_id = schema_id.removesuffix(".json")
    return schema_id


def schema_model_for(schema_id: str) -> type[BaseModel]:
    schema_id = normalize_schema_id(schema_id)
    try:
        return SCHEMA_MODELS[schema_id]
    except KeyError as exc:
        known = ", ".join(sorted(SCHEMA_MODELS))
        raise KeyError(f"Unknown schema_id={schema_id!r}. Known schemas: {known}") from exc


def schema_json_for(schema_id: str) -> dict[str, Any]:
    return schema_model_for(schema_id).model_json_schema()


def validate_output(schema_id: str, output: dict[str, Any] | None) -> tuple[bool, str | None, dict[str, Any] | None]:
    if output is None:
        return False, "No parsed output to validate", None
    try:
        model = schema_model_for(schema_id).model_validate(output)
    except (KeyError, ValidationError) as exc:
        return False, str(exc), None
    return True, None, model.model_dump()

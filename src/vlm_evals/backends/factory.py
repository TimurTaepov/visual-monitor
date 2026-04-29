from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_evals.backends.base import VisionModelBackend
from vlm_evals.backends.mock_backend import MockBackend
from vlm_evals.backends.onnx_backend import ONNXBackend
from vlm_evals.backends.provider_backend import ProviderBackend
from vlm_evals.utils.config import load_yaml


def load_backend_config(path_or_config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_config, dict):
        return dict(path_or_config)
    config = load_yaml(path_or_config)
    config["_config_path"] = str(path_or_config)
    return config


def create_backend(path_or_config: str | Path | dict[str, Any]) -> VisionModelBackend:
    config = load_backend_config(path_or_config)
    backend_type = str(config.get("backend", "")).lower()
    if backend_type == "mock":
        return MockBackend(config)
    if backend_type == "onnx":
        return ONNXBackend(config)
    if backend_type == "provider":
        return ProviderBackend(config)
    raise ValueError(f"Unknown backend type {backend_type!r}")

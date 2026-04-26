from __future__ import annotations

from typing import Any

from vlm_evals.backends.base import VisionModelBackend
from vlm_evals.tasks.schemas import EvalTask


class HFBackend(VisionModelBackend):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._load_error: str | None = None
        try:
            import transformers  # noqa: F401
        except Exception as exc:
            self._load_error = str(exc)

    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if self._load_error:
            raise RuntimeError(
                "Transformers is not available. Install the hf extra with `pip install -e .[hf]`. "
                f"Import error: {self._load_error}"
            )
        raise NotImplementedError(
            "Generic HF VLM inference is model-family-specific. Use vLLM for supported models, "
            "or implement a model-specific adapter in HFBackend before running this config."
        )


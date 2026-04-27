from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from vlm_evals.backends.base import VisionModelBackend
from vlm_evals.backends.openai_compatible import OpenAICompatibleVisionClient
from vlm_evals.tasks.schemas import EvalTask


class ProviderBackend(VisionModelBackend):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        provider = str(config.get("provider", "openai"))
        if provider != "openai":
            raise ValueError(f"Provider backend currently supports openai only, got {provider!r}")
        api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
        client_kwargs = {"api_key": api_key}
        if config.get("base_url"):
            client_kwargs["base_url"] = str(config["base_url"])
        self.client = OpenAICompatibleVisionClient(
            client=OpenAI(**client_kwargs),
            config=self.config,
            request_model_name=self.model_name,
            backend_name="provider",
            reported_model_name=self.model_name,
        )

    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self.client.predict(task, prompt, schema)

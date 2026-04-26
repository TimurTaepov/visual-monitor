from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from vlm_evals.backends.base import VisionModelBackend, prediction_record
from vlm_evals.tasks.schemas import EvalTask
from vlm_evals.utils.cost import estimate_request_cost
from vlm_evals.utils.image import encode_image_data_url
from vlm_evals.utils.timing import Timer


class VLLMBackend(VisionModelBackend):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        api_key_env = str(config.get("api_key_env", "VLLM_API_KEY"))
        api_key = os.getenv(api_key_env) or "EMPTY"
        self.client = OpenAI(base_url=str(config["base_url"]), api_key=api_key)

    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        image_payload = encode_image_data_url(task.image_path)
        with Timer() as timer:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_payload}},
                        ],
                    }
                ],
                temperature=float(self.config.get("temperature", 0.0)),
                max_tokens=int(self.config.get("max_tokens", 256)),
                timeout=float(self.config.get("timeout_seconds", 60)),
            )
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        return prediction_record(
            task=task,
            backend="vllm",
            model_name=self.model_name,
            raw_output=raw,
            latency_ms=timer.elapsed_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=estimate_request_cost(self.config, tokens_in, tokens_out),
        )


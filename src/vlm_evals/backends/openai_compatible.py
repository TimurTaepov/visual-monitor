from __future__ import annotations

from typing import Any

from vlm_evals.backends.base import prediction_record
from vlm_evals.tasks.schemas import EvalTask
from vlm_evals.utils.cost import estimate_request_cost
from vlm_evals.utils.image import encode_image_data_url
from vlm_evals.utils.timing import Timer


class OpenAICompatibleVisionClient:
    def __init__(
        self,
        *,
        client: Any,
        config: dict[str, Any],
        request_model_name: str,
        backend_name: str,
        reported_model_name: str,
    ):
        self.client = client
        self.config = config
        self.request_model_name = request_model_name
        self.backend_name = backend_name
        self.reported_model_name = reported_model_name

    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        image_payload = encode_image_data_url(task.image_path)
        with Timer() as timer:
            response = self.client.chat.completions.create(
                model=self.request_model_name,
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
            backend=self.backend_name,
            model_name=self.reported_model_name,
            raw_output=raw,
            latency_ms=timer.elapsed_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=estimate_request_cost(self.config, tokens_in, tokens_out),
        )

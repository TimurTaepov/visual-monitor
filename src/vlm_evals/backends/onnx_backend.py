from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlm_evals.backends.base import VisionModelBackend, prediction_record
from vlm_evals.tasks.schemas import EvalTask
from vlm_evals.utils.timing import Timer


def _load_onnxruntime_genai() -> Any:
    try:
        import onnxruntime_genai as og
    except ImportError as exc:
        raise RuntimeError(
            "ONNX backend requires onnxruntime-genai. Install it with "
            "`pip install -e '.[onnx]'` for CPU, or install the provider-specific "
            "onnxruntime-genai package for CUDA/DirectML."
        ) from exc
    return og


def _model_path(config: dict[str, Any]) -> Path:
    value = config.get("model_path")
    if not value:
        raise ValueError("ONNX backend requires model_path pointing at an exported model directory")
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ONNX model_path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"ONNX model_path must be a directory: {path}")
    return path


def _execution_provider_name(value: str) -> str:
    aliases = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "dml": "DmlExecutionProvider",
        "directml": "DmlExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "follow_config": "follow_config",
    }
    return aliases.get(value.lower(), value)


def _execution_providers(config: dict[str, Any]) -> list[str]:
    providers = config.get("execution_providers")
    if providers is None:
        provider = config.get("execution_provider")
        if provider is None:
            return []
        providers = [provider]
    if not isinstance(providers, list):
        raise TypeError("execution_providers must be a list")
    return [_execution_provider_name(str(provider)) for provider in providers]


def _image_prompt_content(model_type: str, image_count: int, prompt: str) -> Any:
    if model_type == "phi3v":
        image_tags = "".join(f"<|image_{idx + 1}|>\n" for idx in range(image_count))
        return image_tags + prompt
    if model_type in {"qwen2_5_vl", "qwen2_vl", "qwen3_vl", "fara"}:
        image_tags = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in range(image_count))
        return image_tags + prompt
    return [{"type": "image"} for _ in range(image_count)] + [{"type": "text", "text": prompt}]


def _search_options(config: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if "max_length" in config:
        options["max_length"] = int(config["max_length"])
    if "temperature" in config:
        options["temperature"] = float(config["temperature"])
    if "top_p" in config:
        options["top_p"] = float(config["top_p"])
    if "top_k" in config:
        options["top_k"] = int(config["top_k"])
    if "repetition_penalty" in config:
        options["repetition_penalty"] = float(config["repetition_penalty"])
    if not options:
        options["max_length"] = 7680
    return options


class ONNXBackend(VisionModelBackend):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model_path = _model_path(config)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.stream = None
        self.og = None

    def start(self) -> None:
        if self.model is not None:
            return

        og = _load_onnxruntime_genai()
        providers = _execution_providers(self.config)
        if providers and providers != ["follow_config"]:
            runtime_config = og.Config(str(self.model_path))
            runtime_config.clear_providers()
            for provider in providers:
                runtime_config.append_provider(provider)
            model = og.Model(runtime_config)
        else:
            model = og.Model(str(self.model_path))

        self.og = og
        self.model = model
        self.tokenizer = og.Tokenizer(model)
        self.processor = model.create_multimodal_processor()
        self.stream = self.processor.create_stream()

    def close(self) -> None:
        self.stream = None
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.og = None

    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            self.start()

        assert self.og is not None
        assert self.model is not None
        assert self.processor is not None
        assert self.stream is not None
        assert self.tokenizer is not None

        with Timer() as timer:
            raw = self._generate(task, prompt)

        return prediction_record(
            task=task,
            backend="onnx",
            model_name=self.model_name,
            raw_output=raw,
            latency_ms=timer.elapsed_ms,
            tokens_in=0,
            tokens_out=0,
            estimated_cost_usd=0.0,
        )

    def _generate(self, task: EvalTask, prompt: str) -> str:
        image_paths = [str(Path(path).expanduser()) for path in task.image_paths]
        for image_path in image_paths:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

        images = self.og.Images.open(*image_paths) if image_paths else None
        model_type = str(getattr(self.model, "type", "") or "")
        messages = [
            {
                "role": "user",
                "content": _image_prompt_content(model_type, len(image_paths), prompt),
            }
        ]
        onnx_prompt = self.tokenizer.apply_chat_template(
            json.dumps(messages),
            add_generation_prompt=True,
        )
        inputs = self.processor(onnx_prompt, images=images)
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(**_search_options(self.config))
        generator = self.og.Generator(self.model, params)
        generator.set_inputs(inputs)

        chunks: list[str] = []
        try:
            while not generator.is_done():
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                chunks.append(self.stream.decode(token))
        finally:
            del generator
        return "".join(chunks).strip()

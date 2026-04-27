from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openai import OpenAI

from vlm_evals.backends.base import VisionModelBackend
from vlm_evals.backends.openai_compatible import OpenAICompatibleVisionClient
from vlm_evals.tasks.schemas import EvalTask


def _string_list(values: list[Any], field_name: str) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list")
    return [str(value) for value in values]


def _sanitize_for_path(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def build_vllm_serve_command(config: dict[str, Any]) -> list[str]:
    serve_config = config.get("serve", {}) or {}
    model_name = str(config["model_name"])
    executable = str(serve_config.get("executable", "vllm"))
    host = str(serve_config.get("host", "127.0.0.1"))
    port = str(serve_config.get("port", 8000))

    command = [
        executable,
        "serve",
        model_name,
        "--host",
        host,
        "--port",
        port,
    ]

    served_model_name = serve_config.get("served_model_name") or config.get("served_model_name")
    if served_model_name:
        command.extend(["--served-model-name", str(served_model_name)])

    server_api_key = _server_api_key(config)
    if server_api_key:
        command.extend(["--api-key", server_api_key])

    command.extend(_string_list(serve_config.get("args", []), "serve.args"))
    return command


def _server_api_key(config: dict[str, Any]) -> str | None:
    serve_config = config.get("serve", {}) or {}
    api_key_env = serve_config.get("api_key_env")
    if api_key_env:
        return os.getenv(str(api_key_env))
    return None


class ManagedVLLMServer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.serve_config = config.get("serve", {}) or {}
        self.enabled = bool(self.serve_config.get("enabled", False))
        self.host = str(self.serve_config.get("host", "127.0.0.1"))
        self.port = int(self.serve_config.get("port", 8000))
        self.server_url = str(self.serve_config.get("server_url") or f"http://{self.host}:{self.port}")
        self.base_url = str(config.get("base_url") or f"{self.server_url}/v1")
        self.health_path = str(self.serve_config.get("health_path", "/health"))
        self.startup_timeout_seconds = float(self.serve_config.get("startup_timeout_seconds", 600))
        self.shutdown_timeout_seconds = float(self.serve_config.get("shutdown_timeout_seconds", 30))
        self.poll_interval_seconds = float(self.serve_config.get("poll_interval_seconds", 1))
        self.reuse_existing = bool(self.serve_config.get("reuse_existing", False))
        self.process: subprocess.Popen[str] | None = None
        self.log_file = None
        self._owns_process = False

    def start(self) -> None:
        if not self.enabled:
            return

        if self._is_ready():
            if self.reuse_existing:
                return
            raise RuntimeError(
                f"vLLM endpoint is already responding at {self.server_url}. "
                "Set serve.reuse_existing: true to use it, or choose another port."
            )

        command = build_vllm_serve_command(self.config)
        log_path = self._log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path.open("a", encoding="utf-8")

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in (self.serve_config.get("env", {}) or {}).items()})

        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        self._owns_process = True
        self._wait_until_ready()

    def close(self) -> None:
        if not self.process or not self._owns_process:
            self._close_log()
            return

        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=self.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)

        self._close_log()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    "vLLM server exited before becoming ready. "
                    f"Log tail:\n{self._log_tail()}"
                )
            if self._is_ready():
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(
            f"Timed out waiting {self.startup_timeout_seconds}s for vLLM at {self.server_url}. "
            f"Log tail:\n{self._log_tail()}"
        )

    def _is_ready(self) -> bool:
        url = f"{self.server_url}{self.health_path}"
        request = urllib.request.Request(url)
        api_key = _server_api_key(self.config)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return 200 <= int(response.status) < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def _log_path(self) -> Path:
        configured = self.serve_config.get("log_path")
        if configured:
            return Path(str(configured))
        model_slug = _sanitize_for_path(str(self.config["model_name"]))
        return Path("logs") / f"vllm_{model_slug}_{self.port}.log"

    def _log_tail(self, max_chars: int = 4000) -> str:
        path = self._log_path()
        if not path.exists():
            return "<no vLLM log file found>"
        content = path.read_text(encoding="utf-8", errors="replace")
        return content[-max_chars:] if content else "<empty vLLM log file>"

    def _close_log(self) -> None:
        if self.log_file and not self.log_file.closed:
            self.log_file.close()


class VLLMBackend(VisionModelBackend):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.server = ManagedVLLMServer(config)
        self.request_model_name = str(config.get("served_model_name") or self.model_name)
        self.client: OpenAICompatibleVisionClient | None = None

    def start(self) -> None:
        self.server.start()
        self.client = OpenAICompatibleVisionClient(
            client=OpenAI(base_url=self.server.base_url, api_key=self._client_api_key()),
            config=self.config,
            request_model_name=self.request_model_name,
            backend_name="vllm",
            reported_model_name=self.model_name,
        )

    def close(self) -> None:
        self.server.close()
        self.client = None

    def predict(self, task: EvalTask, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            self.start()
        return self.client.predict(task, prompt, schema)

    def _client_api_key(self) -> str:
        server_api_key = _server_api_key(self.config)
        if server_api_key:
            return server_api_key
        api_key_env = self.config.get("api_key_env")
        if api_key_env:
            return os.getenv(str(api_key_env)) or "EMPTY"
        return "EMPTY"

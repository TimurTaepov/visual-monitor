from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from vlm_evals.backends import create_backend, load_backend_config
from vlm_evals.backends.base import VisionModelBackend


class _BackendHandle:
    def __init__(self, backend: VisionModelBackend):
        self.backend = backend
        self.lock = threading.RLock()


class ServingBackendManager:
    def __init__(self):
        self._handles: dict[str, _BackendHandle] = {}
        self._lock = threading.RLock()

    @contextmanager
    def backend(self, path_or_config: str | Path | dict[str, Any]) -> Iterator[VisionModelBackend]:
        handle = self._get_or_start(path_or_config)
        with handle.lock:
            yield handle.backend

    def close_all(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()

        for handle in handles:
            with handle.lock:
                handle.backend.close()

    def _get_or_start(self, path_or_config: str | Path | dict[str, Any]) -> _BackendHandle:
        key = self._cache_key(path_or_config)
        with self._lock:
            handle = self._handles.get(key)
            if handle is not None:
                return handle

            backend = create_backend(path_or_config)
            try:
                backend.start()
            except Exception:
                backend.close()
                raise
            handle = _BackendHandle(backend)
            self._handles[key] = handle
            return handle

    @staticmethod
    def _cache_key(path_or_config: str | Path | dict[str, Any]) -> str:
        if isinstance(path_or_config, dict):
            return json.dumps(load_backend_config(path_or_config), sort_keys=True, default=str)
        path = Path(path_or_config).expanduser()
        if path.exists():
            return str(path.resolve())
        return str(path_or_config)


backend_manager = ServingBackendManager()

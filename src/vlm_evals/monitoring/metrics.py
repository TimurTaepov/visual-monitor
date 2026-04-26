from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServiceCounters:
    request_count: int = 0
    error_count: int = 0
    timeout_count: int = 0

    def record_request(self, error: bool = False, timeout: bool = False) -> None:
        self.request_count += 1
        self.error_count += int(error)
        self.timeout_count += int(timeout)

    def snapshot(self) -> dict[str, int]:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
        }


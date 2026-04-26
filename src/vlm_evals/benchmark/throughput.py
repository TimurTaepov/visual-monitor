from __future__ import annotations


def requests_per_second(num_requests: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return round(num_requests / elapsed_seconds, 4)


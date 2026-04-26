from __future__ import annotations

from vlm_evals.eval.metrics import percentile


def latency_summary(latencies_ms: list[float]) -> dict[str, float | None]:
    return {
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "p99_latency_ms": percentile(latencies_ms, 0.99),
    }


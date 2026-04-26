from __future__ import annotations

import argparse
import json
import time

from vlm_evals.benchmark.latency import latency_summary
from vlm_evals.benchmark.throughput import requests_per_second
from vlm_evals.eval.run_eval import run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    latencies: list[float] = []
    start = time.perf_counter()
    for _ in range(args.repeats):
        summary = run_eval(args.config)
        for result in summary["results"]:
            latencies.extend(
                float(pred.get("latency_ms", 0.0) or 0.0) for pred in result["predictions"]
            )
    elapsed = time.perf_counter() - start
    output = {
        **latency_summary(latencies),
        "requests_per_second": requests_per_second(len(latencies), elapsed),
        "num_requests": len(latencies),
        "elapsed_seconds": round(elapsed, 3),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


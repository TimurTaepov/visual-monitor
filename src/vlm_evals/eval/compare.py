from __future__ import annotations

import argparse
from pathlib import Path

from vlm_evals.eval.run_eval import run_eval
from vlm_evals.tasks.loader import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/onnx_matrix.yaml")
    args = parser.parse_args()
    summary = run_eval(args.config)
    rows = []
    for result in summary["results"]:
        info = result["backend_info"]
        metrics = result["metrics"]
        rows.append(
            {
                "backend": info["backend"],
                "model_name": info["model_name"],
                "accuracy": metrics.get("accuracy"),
                "valid_json_rate": metrics.get("valid_json_rate"),
                "schema_valid_rate": metrics.get("schema_valid_rate"),
                "hallucination_rate": metrics.get("hallucination_rate"),
                "p95_latency_ms": metrics.get("p95_latency_ms"),
                "review_rate": metrics.get("review_rate"),
                "report_path": result["report_path"],
            }
        )
    output = Path("reports/model_comparison.jsonl")
    write_jsonl(output, rows)
    print(f"Wrote comparison summary to {output}")


if __name__ == "__main__":
    main()

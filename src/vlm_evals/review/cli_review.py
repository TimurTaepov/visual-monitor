from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/local_mock_predictions.jsonl")
    args = parser.parse_args()
    path = Path(args.predictions)
    if not path.exists():
        raise SystemExit(f"Prediction file not found: {path}. Run `make eval` first.")

    total = 0
    review = 0
    reasons: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if record.get("requires_human_review"):
                review += 1
                for reason in record.get("review_reasons", []):
                    reasons[reason] = reasons.get(reason, 0) + 1

    print(f"Predictions: {total}")
    print(f"Review cases: {review}")
    for reason, count in sorted(reasons.items()):
        print(f"- {reason}: {count}")


if __name__ == "__main__":
    main()


#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m vlm_evals.benchmark.load_test --config "${1:-configs/eval/default.yaml}"


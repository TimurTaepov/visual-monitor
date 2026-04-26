#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m vlm_evals.eval.run_eval --config "${1:-configs/eval/default.yaml}"


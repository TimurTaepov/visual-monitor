#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m vlm_evals.eval.compare --config "${1:-configs/eval/small_vlm_matrix.yaml}"


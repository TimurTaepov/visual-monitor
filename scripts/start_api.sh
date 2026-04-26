#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src uvicorn vlm_evals.serving.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"


# Visual LLM Evaluation and Serving Benchmark

A production-style benchmark for evaluating and serving small visual language models on operational monitoring tasks.

The first runnable slice includes:

- JSONL task loading and prompt rendering
- schema validation for structured model outputs
- managed vLLM serving for Hugging Face model ids
- provider API backend for hosted baselines
- deterministic local smoke-test backend for harness verification
- quality, JSON reliability, hallucination, review routing, latency, and cost proxy metrics
- Markdown reports and JSONL prediction artifacts
- FastAPI serving surface for prediction and evaluation

## Quickstart

```bash
make eval
make test
```

The default config is a smoke test that does not download models or call provider APIs.
Use `make eval-vllm` or `make compare` for managed vLLM-backed evaluation.

## Real Model Path

1. Put dataset files under `data/raw/`.
2. Convert raw annotations into `data/tasks/*.jsonl`.
3. Install vLLM in the runtime environment.
4. Point an eval config at `configs/backends/*_vllm.yaml`.
5. Run `make eval-vllm` or `make compare`.

For `backend: vllm` configs with `serve.enabled: true`, the evaluator starts `vllm serve`
for the configured Hugging Face model id, waits for readiness, runs the evaluation, and
shuts the server down at the end of the backend run.

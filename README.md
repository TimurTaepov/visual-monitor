# Visual LLM Evaluation and Serving Benchmark

A production-style benchmark for evaluating and serving small visual language models on operational monitoring tasks.

The first runnable slice includes:

- JSONL task loading and prompt rendering
- schema validation for structured model outputs
- deterministic mock backend for local verification
- pluggable HF, vLLM, provider, and mock backend registry
- quality, JSON reliability, hallucination, review routing, latency, and cost proxy metrics
- Markdown reports and JSONL prediction artifacts
- FastAPI serving surface for prediction and evaluation

## Quickstart

```bash
make eval
make test
```

The default config uses the mock backend, so it does not download models or call provider APIs.

## Real Model Path

1. Put dataset files under `data/raw/`.
2. Convert raw annotations into `data/tasks/*.jsonl`.
3. Start a supported vLLM server, for example Qwen3-VL-2B.
4. Point an eval config at `configs/backends/*_vllm.yaml`.
5. Run `make compare`.

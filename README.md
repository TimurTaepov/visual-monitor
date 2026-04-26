# Visual LLM Evaluation and Serving Benchmark

Evaluate small vision-language models on image tasks, compare their outputs, and measure the basics you need before trusting one in an application.

The repo currently supports:

- evaluation tasks stored as JSONL
- prompt files and JSON/Pydantic schemas
- local vLLM serving for Hugging Face model ids
- optional provider API backends
- a small local check that runs without models or API keys
- reports with accuracy, JSON validity, review routing, and latency numbers
- a FastAPI wrapper for `/predict` and `/evaluate`

## Quickstart

Run the local check first:

```bash
make eval
make test
```

This uses `configs/backends/mock_oracle.yaml`, so it does not download a model or call an API. It only checks that task loading, prompt rendering, validation, scoring, and report writing work.

## Run a model with vLLM

1. Install vLLM in the environment where you will run the evaluation.
2. Put images and labels into `data/tasks/*.jsonl`.
3. Check the model config in `configs/backends/*_vllm.yaml`.
4. Run:

```bash
make eval-vllm
```

For each vLLM config, the evaluator starts `vllm serve <model_name>`, waits for the server to answer `/health`, sends the image+prompt requests, writes the report, and then stops the vLLM process.

To compare several model configs, run:

```bash
make compare
```

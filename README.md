# Visual LLM Evaluation and Serving Benchmark

Evaluate small vision-language models on image tasks, compare their outputs, and measure the basics you need before trusting one in an application.

The repo currently supports:

- evaluation tasks stored as JSONL
- prompt files and JSON/Pydantic schemas
- one-image and multi-image eval tasks
- local vLLM serving for Hugging Face model ids
- optional provider API backends
- dataset adapters for benchmark-specific formats, with VLM-SubtleBench included as an example
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

## Example: run VLM-SubtleBench

This is one example of a dataset-specific benchmark adapter. The core eval runner is not tied to SubtleBench; it can still run local JSONL tasks or other adapters.

Download the dataset from Hugging Face into `VLM-SubtleBench/`.

Then choose a backend config. The benchmark does not care whether the model is behind Together, another OpenAI-compatible provider, or a vLLM server. The eval config decides that through its `backends` list.

The default command uses the Together example config:

```bash
make eval-subtlebench
```

To run the same benchmark through a vLLM-served model:

```bash
make eval-subtlebench SUBTLEBENCH_CONFIG=configs/eval/subtlebench_vllm.yaml
```

The example config loads paired-image questions, sends both images to the chosen backend, and scores the returned `answer` field against the benchmark label.

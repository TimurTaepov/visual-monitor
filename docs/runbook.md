# Runbook

## Check the eval code

```bash
make eval
make test
```

This is the fastest sanity check. It uses the local mock backend, so it does not need downloaded weights or API keys.

Expected output:

- `make eval` writes a report under `reports/`
- `make test` runs the unit tests

## API

```bash
make serve
```

Then visit:

- `GET /health`
- `GET /models`
- `GET /tasks`

The API keeps model backends alive between requests. It closes them when the FastAPI process shuts down.

## Run a local ONNX model

Use this when you want the eval runner to load an exported ONNX Runtime GenAI model from disk.

First check the backend config:

```text
configs/backends/phi3_5_vision_onnx.yaml
```

The important fields are:

- `model_name`: name written to reports
- `model_path`: local directory containing the exported ONNX Runtime GenAI model
- `execution_provider`: `follow_config`, `cpu`, `cuda`, `directml`, or a provider name
- `max_length`: generation length passed to ONNX Runtime GenAI

Then run:

```bash
make eval-onnx
```

What happens:

1. The eval runner loads the ONNX model once.
2. It sends each image and prompt to the local model.
3. It parses the model output as JSON.
4. It scores the output against the task labels.
5. It writes predictions and a Markdown report.
6. It releases the model.

For CPU, install:

```bash
pip install -e '.[onnx]'
```

For CUDA or DirectML, install the matching ONNX Runtime GenAI package for that provider.

## Run VLM-SubtleBench

Download the dataset first. It is large, so keep it outside git:

```bash
HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('KRAFTON/VLM-SubtleBench', repo_type='dataset', local_dir='VLM-SubtleBench', max_workers=1)"
```

The benchmark setup and the model backend are separate:

- the dataset block tells the runner how to load VLM-SubtleBench
- the scoring block tells the runner how to score multiple-choice answers
- the `backends` list tells the runner which model backend to call

The default command uses the Together example config:

```bash
make eval-subtlebench
```

That command reads:

```text
configs/eval/subtlebench_together.yaml
```

To run the same benchmark through a local ONNX model:

```bash
make eval-subtlebench SUBTLEBENCH_CONFIG=configs/eval/subtlebench_onnx.yaml
```

To use another API provider, create or edit a backend config and point `backends` at it:

```yaml
backends:
  - configs/backends/my_provider_model.yaml
```

For an OpenAI-compatible provider, the backend config should look like:

```yaml
backend: provider
provider: openai
model_name: provider/model-name
base_url: https://provider.example.com/v1
api_key_env: PROVIDER_API_KEY
temperature: 0.0
max_tokens: 256
timeout_seconds: 90
```

For Together, use `provider: together`; the base URL and `TOGETHER_API_KEY` default are already handled.

For ONNX, use `backend: onnx` and point `model_path` at an exported ONNX Runtime GenAI model directory.

What the SubtleBench adapter does:

1. Reads the metadata from `VLM-SubtleBench/data/test.jsonl`, `VLM-SubtleBench/data/test.json`, or `qa.json`.
2. Creates one eval task per image pair.
3. Sends both images to the configured backend in order.
4. Asks the model to return JSON with `answer`, `confidence`, `evidence`, and `requires_review`.
5. Scores `answer` against the benchmark label.
6. Writes predictions and a report under `reports/subtlebench/`, including category and domain breakdowns.

Useful config fields:

- `dataset.split`: `test`, `val`, or `all`
- `dataset.category`: one category or a list, for example `state`
- `dataset.domain`: one domain or a list, for example `industrial`
- `dataset.max_examples`: set this for a small check before running the full split
- `dataset.skip_missing_images`: keeps the run moving when local medical images are not present
- `backends`: the model config or configs to evaluate

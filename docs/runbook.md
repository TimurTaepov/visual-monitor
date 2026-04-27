# Runbook

## Check the eval code

```bash
make eval
make test
```

This is the fastest sanity check. It uses the local mock backend, so it does not need vLLM, a GPU, downloaded weights, or API keys.

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

## Run a model with vLLM

Use this when you want vLLM to serve a local model during evaluation.

First check the backend config, for example:

```text
configs/backends/qwen3_vl_2b_vllm.yaml
```

The important fields are:

- `model_name`: the Hugging Face model id passed to `vllm serve`
- `base_url`: where requests will be sent
- `serve.enabled`: whether the eval runner starts and stops vLLM for you
- `serve.args`: extra CLI flags passed to `vllm serve`

Then run:

```bash
make eval-vllm
```

What happens:

1. The eval runner starts `vllm serve <model_name>`.
2. It waits until `/health` responds.
3. It sends each image and prompt to the local vLLM server.
4. It parses the model output as JSON.
5. It scores the output against the task labels.
6. It writes predictions and a Markdown report.
7. It stops the vLLM process.

## Use an already running vLLM server

If you started vLLM yourself, set this in the backend config:

```yaml
serve:
  enabled: false
```

Then point `base_url` at the server:

```yaml
base_url: http://127.0.0.1:8001/v1
```

## Run VLM-SubtleBench

Download the dataset first. It is large, so keep it outside git:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('KRAFTON/VLM-SubtleBench', repo_type='dataset', local_dir='VLM-SubtleBench')"
```

The benchmark setup and the model backend are separate:

- the dataset block tells the runner how to load VLM-SubtleBench
- the scoring block tells the runner how to score multiple-choice answers
- the `backends` list tells the runner which model endpoint to call

The default command uses the Together example config:

```bash
make eval-subtlebench
```

That command reads:

```text
configs/eval/subtlebench_together.yaml
```

To run the same benchmark through vLLM instead:

```bash
make eval-subtlebench SUBTLEBENCH_CONFIG=configs/eval/subtlebench_vllm.yaml
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

For vLLM, use a `backend: vllm` config. If `serve.enabled: true`, the eval runner starts vLLM, waits for it, sends the benchmark requests, then stops it.

What it does:

1. Reads the VLM-SubtleBench metadata from `VLM-SubtleBench/data/test.jsonl`, `VLM-SubtleBench/data/test.json`, or `qa.json`.
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

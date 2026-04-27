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

## Run VLM-SubtleBench on Together

Download the dataset first. It is large, so keep it outside git:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('KRAFTON/VLM-SubtleBench', repo_type='dataset', local_dir='VLM-SubtleBench')"
```

Set the Together key:

```bash
export TOGETHER_API_KEY=...
```

Then run:

```bash
make eval-subtlebench
```

The config is:

```text
configs/eval/subtlebench_together.yaml
```

What it does:

1. Reads the VLM-SubtleBench metadata from `VLM-SubtleBench/data/test.jsonl`, `VLM-SubtleBench/data/test.json`, or `qa.json`.
2. Creates one eval task per image pair.
3. Sends both images to the provider endpoint in order.
4. Asks the model to return JSON with `answer`, `confidence`, `evidence`, and `requires_review`.
5. Scores `answer` against the benchmark label.
6. Writes predictions and a report under `reports/subtlebench/`, including category and domain breakdowns.

Useful config fields:

- `dataset.split`: `test`, `val`, or `all`
- `dataset.category`: one category or a list, for example `state`
- `dataset.domain`: one domain or a list, for example `industrial`
- `dataset.max_examples`: set this for a small check before running the full split
- `dataset.skip_missing_images`: keeps the run moving when local medical images are not present
- `backends`: point this to the Together model config you want to evaluate

To use a different Together-hosted VLM, edit:

```text
configs/backends/together_qwen3_5_9b.yaml
```

and change `model_name`.

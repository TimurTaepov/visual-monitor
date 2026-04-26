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

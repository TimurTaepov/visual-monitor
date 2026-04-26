# Runbook

## Local Harness Check

```bash
make eval
make test
```

## API

```bash
make serve
```

Then visit:

- `GET /health`
- `GET /models`
- `GET /tasks`

## Real vLLM Backend

Managed mode is the default for the checked-in vLLM backend configs.

```bash
make eval-vllm
```

For each `backend: vllm` config with `serve.enabled: true`, the evaluator runs
`vllm serve <model_name>`, waits for `/health`, sends OpenAI-compatible multimodal
chat requests, and terminates the vLLM process when that backend finishes.

To use an already-running vLLM server instead, set `serve.enabled: false` and point
`base_url` at the server's `/v1` endpoint.

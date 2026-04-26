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

Start vLLM separately and point a backend config at its OpenAI-compatible endpoint.

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct --host 0.0.0.0 --port 8001 --dtype auto
```


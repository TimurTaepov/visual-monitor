# Evaluation Methodology

The benchmark evaluates a model/backend/prompt combination across:

- task correctness
- structured output reliability
- hallucination and unsupported evidence checks
- human-review routing quality
- latency and throughput
- cost proxy

The first local run uses a deterministic mock backend to validate the harness. Real model reports should include package versions, hardware, model commit or revision, backend config, and dataset version.


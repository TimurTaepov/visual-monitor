# Model backend notes

Use this file to record which models actually run in the local environment.

| Model | Status | Notes |
|---|---|---|
| mistralai/Ministral-3-3B-Instruct-2512 | not tested | vLLM config exists. |
| Qwen/Qwen3-VL-2B-Instruct | not tested | vLLM config exists. |
| OpenGVLab/InternVL3_5-2B-HF | not tested | vLLM config exists. |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | not tested | vLLM config exists. |
| Qwen/Qwen3.5-9B | not tested | Together config exists. Replace with another Together vision model when needed. |

When testing a model, add:

- machine/GPU
- vLLM version
- model revision, if pinned
- whether startup succeeded
- whether image requests worked
- any special flags needed in `serve.args`

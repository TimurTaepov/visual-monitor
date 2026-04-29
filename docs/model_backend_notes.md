# Model backend notes

Use this file to record which models actually run in the local environment.

| Model | Backend | Status | Notes |
|---|---|---|---|
| microsoft/Phi-3.5-vision-instruct-onnx | ONNX Runtime GenAI | not tested | Example config exists. Download an ONNX export and set `model_path`. |
| google/gemma-3n-E4B-it | Together provider | smoke tested | 30-example SubtleBench run completed through Together. |
| Qwen/Qwen3.5-9B | Together provider | not tested | Replace with another provider vision model when needed. |

When testing a local ONNX model, record:

- machine/GPU
- ONNX Runtime GenAI package and version
- execution provider
- model export path and model revision
- whether startup succeeded
- whether one-image and two-image requests worked

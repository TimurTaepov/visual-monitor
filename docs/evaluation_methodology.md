# Evaluation methodology

Each run evaluates one or more model configs against the same task set.

Tasks can come from a local JSONL file or from a dataset adapter such as VLM-SubtleBench.

Each task row includes:

- one image path or multiple image paths
- task type
- prompt template
- expected schema
- labels
- metadata

For each task, the runner sends the images and prompt to the backend, parses the answer, validates the JSON, and scores the configured prediction field against the configured label field.

For VLM-SubtleBench multiple-choice runs:

- each task contains two image paths
- metadata includes the benchmark question, choices, category, domain, source, and optional caption
- scoring compares the model's `answer` to the benchmark `answer`
- reports include category and domain breakdowns

The report includes:

- task correctness
- valid JSON rate
- schema-valid output rate
- simple checks for wrong, high-confidence claims
- human-review routing reasons
- latency
- estimated cost, when configured

For real model runs, record the model revision, vLLM version, hardware, backend config, and dataset version. Otherwise the numbers will be hard to reproduce.

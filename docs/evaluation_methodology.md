# Evaluation methodology

Each run evaluates one or more model configs against the same task file.

Each task row includes:

- image path
- task type
- prompt template
- expected schema
- labels
- metadata

For each task, the runner sends the image and prompt to the backend, parses the answer, validates the JSON, and compares the answer to the label.

The report includes:

- task correctness
- valid JSON rate
- schema-valid output rate
- simple checks for wrong, high-confidence claims
- human-review routing reasons
- latency
- estimated cost, when configured

For real model runs, record the model revision, vLLM version, hardware, backend config, and dataset version. Otherwise the numbers will be hard to reproduce.

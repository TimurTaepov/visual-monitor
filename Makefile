.PHONY: setup eval compare serve benchmark review test format lint

PYTHONPATH := src

setup:
	pip install -e .

eval:
	PYTHONPATH=$(PYTHONPATH) python -m vlm_evals.eval.run_eval --config configs/eval/default.yaml

compare:
	PYTHONPATH=$(PYTHONPATH) python -m vlm_evals.eval.compare --config configs/eval/default.yaml

serve:
	PYTHONPATH=$(PYTHONPATH) uvicorn vlm_evals.serving.app:app --host 0.0.0.0 --port 8000

benchmark:
	PYTHONPATH=$(PYTHONPATH) python -m vlm_evals.benchmark.load_test --config configs/eval/default.yaml

review:
	PYTHONPATH=$(PYTHONPATH) python -m vlm_evals.review.cli_review

test:
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s tests

format:
	ruff format src tests

lint:
	ruff check src tests


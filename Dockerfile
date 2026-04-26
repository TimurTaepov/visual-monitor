FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir -e .
COPY . /app

ENV PYTHONPATH=/app/src
CMD ["uvicorn", "vlm_evals.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]


from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from vlm_evals.tasks.schemas import EvalTask

SUBTLEBENCH_TASK_TYPE = "subtlebench_multiple_choice"
SUBTLEBENCH_SCHEMA = "subtlebench_multiple_choice_schema"
SUBTLEBENCH_PROMPT = "subtlebench_multiple_choice_v1"


def load_subtlebench_tasks(config: dict[str, Any]) -> list[EvalTask]:
    dataset_root = Path(str(config.get("path", "VLM-SubtleBench")))
    split = config.get("split", "test")
    rows = list(_load_rows(dataset_root, None if split in (None, "all") else str(split)))

    category = _optional_filter(config.get("category"))
    domain = _optional_filter(config.get("domain"))
    source = _optional_filter(config.get("source"))
    has_caption_only = bool(config.get("has_caption_only", False))
    skip_missing_images = bool(config.get("skip_missing_images", True))
    max_examples = config.get("max_examples")

    tasks: list[EvalTask] = []
    for idx, row in enumerate(rows):
        if category and str(row.get("category", "")).lower() not in category:
            continue
        if domain and str(row.get("domain", "")).lower() not in domain:
            continue
        if source and str(row.get("source", "")).lower() not in source:
            continue
        if has_caption_only and not bool(row.get("has_caption")):
            continue

        image_paths = _image_paths(row, dataset_root)
        if skip_missing_images and not _images_available(image_paths):
            continue

        tasks.append(_row_to_task(row, image_paths, idx, config))
        if max_examples is not None and len(tasks) >= int(max_examples):
            break

    return tasks


def _load_rows(dataset_root: Path, split: str | None) -> Iterable[dict[str, Any]]:
    if split is None:
        split_files = [
            path
            for split_name in ("test", "val")
            for path in _candidate_files(dataset_root, split_name)
            if path.exists() and path.name.startswith(split_name)
        ]
        if split_files:
            for path in split_files:
                yield from _read_rows(path, None)
            return

    for path in _candidate_files(dataset_root, split):
        if path.exists():
            yield from _read_rows(path, split)
            return
    tried = ", ".join(str(path) for path in _candidate_files(dataset_root, split))
    raise FileNotFoundError(f"Could not find VLM-SubtleBench metadata. Tried: {tried}")


def _candidate_files(dataset_root: Path, split: str | None) -> list[Path]:
    names: list[str] = []
    if split:
        names.extend([f"{split}.jsonl", f"{split}.json"])
    names.extend(["qa.jsonl", "qa.json"])
    return [dataset_root / "data" / name for name in names] + [dataset_root / name for name in names]


def _read_rows(path: Path, split: str | None) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if _split_matches(row, split):
                        yield row
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif split and isinstance(data, dict) and isinstance(data.get(split), list):
        rows = data[split]
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        rows = data["data"]
    elif isinstance(data, dict) and isinstance(data.get("questions"), list):
        rows = data["questions"]
    else:
        raise ValueError(f"Unsupported VLM-SubtleBench metadata shape in {path}")

    for row in rows:
        if not isinstance(row, dict):
            continue
        if _split_matches(row, split):
            yield row


def _split_matches(row: dict[str, Any], split: str | None) -> bool:
    return split is None or "split" not in row or str(row.get("split")) == split


def _optional_filter(value: Any) -> set[str]:
    if value in (None, "", "all"):
        return set()
    if isinstance(value, list):
        return {str(item).lower() for item in value}
    return {str(value).lower()}


def _image_paths(row: dict[str, Any], dataset_root: Path) -> list[str]:
    image_1 = _image_value(row, ["image_1", "first_image", "image1", "image_1_path"])
    image_2 = _image_value(row, ["image_2", "second_image", "image2", "image_2_path"])
    if not image_1 or not image_2:
        raise ValueError(f"SubtleBench row is missing image_1/image_2 fields: {row}")
    return [_resolve_image_path(image_1, dataset_root), _resolve_image_path(image_2, dataset_root)]


def _image_value(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for nested_key in ("path", "url", "src"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested:
                    return nested
    return None


def _resolve_image_path(value: str, dataset_root: Path) -> str:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        marker = "/resolve/main/"
        if marker in parsed.path:
            relative = parsed.path.split(marker, 1)[1]
            local_path = dataset_root / relative
            if local_path.exists():
                return str(local_path)
        return value

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(dataset_root / path)


def _images_available(image_paths: list[str]) -> bool:
    for image_path in image_paths:
        parsed = urlparse(image_path)
        if parsed.scheme in {"http", "https", "data"}:
            continue
        if not Path(image_path).exists():
            return False
    return True


def _row_to_task(
    row: dict[str, Any],
    image_paths: list[str],
    idx: int,
    config: dict[str, Any],
) -> EvalTask:
    answer = str(row["answer"])
    choices = _choices(row)
    source_id = str(row.get("source_id", idx))
    source = str(row.get("source", "unknown"))
    split = str(config.get("split", row.get("split", "all")))
    task_id = str(row.get("id") or f"subtlebench:{split}:{source}:{source_id}")
    metadata = {
        "benchmark": "VLM-SubtleBench",
        "question": row.get("question"),
        "choices": choices,
        "category": row.get("category"),
        "domain": row.get("domain"),
        "source": row.get("source"),
        "source_id": row.get("source_id"),
        "raw_folder": row.get("raw_folder"),
        "has_caption": bool(row.get("has_caption", False)),
        "caption": row.get("caption"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return EvalTask(
        task_id=task_id,
        image_paths=image_paths,
        task_type=SUBTLEBENCH_TASK_TYPE,
        prompt_template=str(config.get("prompt_template", SUBTLEBENCH_PROMPT)),
        expected_schema=str(config.get("expected_schema", SUBTLEBENCH_SCHEMA)),
        labels={"answer": answer},
        metadata=metadata,
    )


def _choices(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if isinstance(row.get("choices"), list):
        values.extend(str(value) for value in row["choices"])
    else:
        values.append(str(row["answer"]))
        distractors = row.get("distractors", [])
        if isinstance(distractors, list):
            values.extend(str(value) for value in distractors)

    seen: set[str] = set()
    deduped = []
    for value in values:
        normalized = _normalize_choice(value)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(value)

    preferred_order = {"first image": 0, "second image": 1}
    return sorted(deduped, key=lambda value: preferred_order.get(_normalize_choice(value), 99))


def _normalize_choice(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())

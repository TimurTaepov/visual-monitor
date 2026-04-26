from __future__ import annotations

import json
import re
from typing import Any

from vlm_evals.tasks.schemas import validate_output


def extract_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_output(raw_output: str | dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool, str | None]:
    if raw_output is None:
        return None, False, "No output"
    if isinstance(raw_output, dict):
        return raw_output, True, None

    json_text = extract_json_object(str(raw_output))
    if json_text is None:
        return None, False, "No JSON object found"
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, False, str(exc)
    if not isinstance(parsed, dict):
        return None, False, "Parsed JSON is not an object"
    return parsed, True, None


def parse_and_validate(
    raw_output: str | dict[str, Any] | None,
    schema_id: str,
) -> dict[str, Any]:
    parsed, valid_json, parse_error = parse_json_output(raw_output)
    schema_valid, schema_error, normalized = validate_output(schema_id, parsed)
    return {
        "parsed_output": normalized if schema_valid else parsed,
        "valid_json": valid_json,
        "schema_valid": schema_valid,
        "parse_error": parse_error,
        "schema_error": schema_error,
    }


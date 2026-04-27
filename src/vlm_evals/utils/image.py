from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse


def encode_image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def image_url_payload(image_path_or_url: str | Path) -> str:
    value = str(image_path_or_url)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "data"}:
        return value
    return encode_image_data_url(value)

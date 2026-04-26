from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def structured_log(event: str, **fields: Any) -> str:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    return json.dumps(payload, sort_keys=True)


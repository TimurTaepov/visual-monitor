from __future__ import annotations

from typing import Any


def estimate_request_cost(config: dict[str, Any], tokens_in: int = 0, tokens_out: int = 0) -> float:
    per_1k_requests = float(config.get("estimated_cost_per_1k_requests_usd", 0.0) or 0.0)
    token_in_price = float(config.get("input_cost_per_1m_tokens_usd", 0.0) or 0.0)
    token_out_price = float(config.get("output_cost_per_1m_tokens_usd", 0.0) or 0.0)
    request_cost = per_1k_requests / 1000.0
    token_cost = (tokens_in / 1_000_000.0) * token_in_price
    token_cost += (tokens_out / 1_000_000.0) * token_out_price
    return round(request_cost + token_cost, 8)


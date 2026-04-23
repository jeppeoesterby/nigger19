"""Token -> USD conversion. Prices loaded from config.yaml."""
from __future__ import annotations


def _coerce(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def cost_usd(model_key: str, input_tokens, output_tokens, pricing: dict) -> float:
    """Compute cost in USD. Robust to None/missing token counts and missing pricing."""
    entry = pricing.get(model_key)
    if not entry:
        return 0.0
    in_tok = _coerce(input_tokens)
    out_tok = _coerce(output_tokens)
    in_price = _coerce(entry.get("input"))
    out_price = _coerce(entry.get("output"))
    return (in_tok / 1_000_000) * in_price + (out_tok / 1_000_000) * out_price

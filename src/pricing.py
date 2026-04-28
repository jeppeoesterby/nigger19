"""Token -> USD conversion. Prices loaded from config.yaml.

Anthropic prompt caching pricing (when used):
  - Regular input tokens: 1.0x
  - Cache creation tokens: 1.25x (one-time write, charged at +25%)
  - Cache read tokens: 0.1x (subsequent reads, 90% discount)

These multipliers come from Anthropic's pricing docs. Gemini does not yet
expose equivalent fields, so cache_creation/cache_read are always 0 there.
"""
from __future__ import annotations


def _coerce(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def cost_usd(
    model_key: str,
    input_tokens,
    output_tokens,
    pricing: dict,
    *,
    cache_creation_tokens=0,
    cache_read_tokens=0,
) -> float:
    """Compute cost in USD.

    Robust to None/missing token counts and missing pricing entries. When
    Anthropic prompt caching is in use, cache_creation_tokens are charged at
    1.25x the input rate and cache_read_tokens at 0.1x.
    """
    entry = pricing.get(model_key)
    if not entry:
        return 0.0
    in_tok = _coerce(input_tokens)
    out_tok = _coerce(output_tokens)
    cc_tok = _coerce(cache_creation_tokens)
    cr_tok = _coerce(cache_read_tokens)
    in_price = _coerce(entry.get("input"))
    out_price = _coerce(entry.get("output"))
    return (
        (in_tok / 1_000_000) * in_price
        + (cc_tok / 1_000_000) * in_price * 1.25
        + (cr_tok / 1_000_000) * in_price * 0.10
        + (out_tok / 1_000_000) * out_price
    )

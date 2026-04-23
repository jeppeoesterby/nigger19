"""Token -> USD conversion. Prices loaded from config.yaml."""
from __future__ import annotations


def cost_usd(model_key: str, input_tokens: int, output_tokens: int, pricing: dict) -> float:
    """Compute cost in USD. pricing is the config.yaml 'pricing' dict."""
    entry = pricing.get(model_key)
    if not entry:
        return 0.0
    return (input_tokens / 1_000_000) * entry["input"] + (
        output_tokens / 1_000_000
    ) * entry["output"]

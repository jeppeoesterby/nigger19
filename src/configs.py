"""The 6 model configurations under test."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    provider: str  # "claude" | "gemini"
    model_key: str  # key into config.yaml models/pricing (e.g. "claude-sonnet-4-6")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    extraction: Stage
    reasoning: Stage

    @property
    def is_hybrid(self) -> bool:
        return (self.extraction.provider, self.extraction.model_key) != (
            self.reasoning.provider,
            self.reasoning.model_key,
        )


CONFIGS: list[ModelConfig] = [
    ModelConfig(
        name="Gemini 3 Pro",
        extraction=Stage("gemini", "gemini-3-pro"),
        reasoning=Stage("gemini", "gemini-3-pro"),
    ),
    ModelConfig(
        name="Gemini 2.5 Pro",
        extraction=Stage("gemini", "gemini-2.5-pro"),
        reasoning=Stage("gemini", "gemini-2.5-pro"),
    ),
    ModelConfig(
        name="Gemini 2.5 Flash",
        extraction=Stage("gemini", "gemini-2.5-flash"),
        reasoning=Stage("gemini", "gemini-2.5-flash"),
    ),
    ModelConfig(
        name="Claude Sonnet 4.6",
        extraction=Stage("claude", "claude-sonnet-4-6"),
        reasoning=Stage("claude", "claude-sonnet-4-6"),
    ),
    ModelConfig(
        name="Claude Opus 4.7",
        extraction=Stage("claude", "claude-opus-4-7"),
        reasoning=Stage("claude", "claude-opus-4-7"),
    ),
    ModelConfig(
        name="Hybrid Sonnet",
        extraction=Stage("gemini", "gemini-3-pro"),
        reasoning=Stage("claude", "claude-sonnet-4-6"),
    ),
    ModelConfig(
        name="Hybrid Opus",
        extraction=Stage("gemini", "gemini-3-pro"),
        reasoning=Stage("claude", "claude-opus-4-7"),
    ),
]


def filter_configs(names: list[str] | None) -> list[ModelConfig]:
    if not names:
        return list(CONFIGS)
    wanted = {n.strip().lower() for n in names}
    picked = [c for c in CONFIGS if c.name.lower() in wanted]
    missing = wanted - {c.name.lower() for c in picked}
    if missing:
        valid = ", ".join(c.name for c in CONFIGS)
        raise ValueError(f"Unknown config(s): {sorted(missing)}. Valid: {valid}")
    return picked

"""Prompt versioning + rollback.

Every accepted autotune suggestion (or manual prompt edit) creates a new
version snapshot. The user can roll back to any prior version with one
click. Keeps the user in the driver's seat: even if an auto-suggested
change turns out to hurt recall, undo is trivial.

Storage: JSONL at ``data/prompt_history.jsonl`` (append-only).
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .prompts import PromptTemplates


@dataclass
class PromptVersion:
    version_id: str
    timestamp: str
    intro: str
    instructions_extraction_only: str
    instructions_with_agreement: str
    reasoning_instructions: str

    # What triggered this version
    source: str = "manual"  # "manual" | "autotune" | "rollback"
    autotune_run_id: Optional[str] = None
    applied_suggestion_ids: list[str] = field(default_factory=list)
    note: Optional[str] = None  # short label, e.g. "tightened varenummer matching"

    def to_templates(self) -> PromptTemplates:
        return PromptTemplates(
            intro=self.intro,
            instructions_extraction_only=self.instructions_extraction_only,
            instructions_with_agreement=self.instructions_with_agreement,
            reasoning_instructions=self.reasoning_instructions,
        )

    @classmethod
    def from_templates(
        cls,
        t: PromptTemplates,
        *,
        source: str = "manual",
        autotune_run_id: Optional[str] = None,
        applied_suggestion_ids: Optional[list[str]] = None,
        note: Optional[str] = None,
    ) -> "PromptVersion":
        return cls(
            version_id=uuid.uuid4().hex[:10],
            timestamp=datetime.now().isoformat(timespec="seconds"),
            intro=t.intro,
            instructions_extraction_only=t.instructions_extraction_only,
            instructions_with_agreement=t.instructions_with_agreement,
            reasoning_instructions=t.reasoning_instructions,
            source=source,
            autotune_run_id=autotune_run_id,
            applied_suggestion_ids=applied_suggestion_ids or [],
            note=note,
        )


class PromptHistory:
    """Append-only versioned store of prompt snapshots."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def list_versions(self) -> list[PromptVersion]:
        if not self.path.exists():
            return []
        out: list[PromptVersion] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(PromptVersion(**json.loads(line)))
                except Exception:
                    continue
        return out

    def latest(self) -> Optional[PromptVersion]:
        versions = self.list_versions()
        if not versions:
            return None
        return max(versions, key=lambda v: v.timestamp)

    def get(self, version_id: str) -> Optional[PromptVersion]:
        for v in self.list_versions():
            if v.version_id == version_id:
                return v
        return None

    def save_snapshot(self, version: PromptVersion) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(version), ensure_ascii=False))
                f.write("\n")

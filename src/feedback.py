"""Feedback capture + storage for the LLM auditor.

Lets the user mark, per (invoice, config) pair, which model-flagged
discrepancies were correct vs false positives, and add discrepancies
the model entirely missed. This is the raw material for prompt
iteration — without ground-truth labels, every prompt change is a
guess. With feedback, we can measure precision/recall per model and
know whether a prompt change actually helped.

Storage: JSONL at ``data/feedback.jsonl`` (one entry per line). Edits
to an existing entry rewrite the whole file; the file is small enough
(thousands of entries max) that this is fine.

This module is a self-contained add-on. The runner and excel writer
do not depend on it; everything still works without any feedback being
captured.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Verdict on a single model-flagged discrepancy
LINE_VERDICTS = (
    "correct",            # model correctly flagged this as a discrepancy
    "false_positive",     # model flagged it but agreement actually allows this price
    "wrong_amount",       # right that there's a discrepancy, but the amount/agreed_price is off
    "no_match_needed",    # model couldn't match — and that's fine (e.g. one-off item not in agreement)
)

OVERALL_VERDICTS = ("good", "acceptable", "poor")

AGREEMENT_TYPES = (
    "price_agreement",     # static price list / fixed prices
    "index_agreement",     # indexed (e.g. quarterly adjustment to BYG-DTU index)
    "project_based",       # project-specific pricing
    "framework",           # framework agreement / rammeaftale
    "other",
)


@dataclass
class LineFeedback:
    """User verdict on one of the model's flagged discrepancies."""

    line_index: int  # which line in the model's line_items array

    # Snapshot of what the model said. Stored inline so feedback survives
    # even if the original run results are deleted from disk.
    model_item_number: Optional[str] = None
    model_description: Optional[str] = None
    model_quantity: Optional[float] = None
    model_unit_price: Optional[float] = None
    model_agreed_unit_price: Optional[float] = None
    model_has_discrepancy: Optional[bool] = None
    model_discrepancy_amount: Optional[float] = None

    # User's verdict
    verdict: str = "correct"  # one of LINE_VERDICTS
    correct_unit_price: Optional[float] = None
    correct_agreed_unit_price: Optional[float] = None
    correct_discrepancy_amount: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class MissedDiscrepancy:
    """A real discrepancy the model entirely failed to flag.

    Captured by the user during review. These are the most valuable
    feedback — they directly indicate prompt-tuning targets.
    """

    item_number: Optional[str] = None
    description: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    correct_agreed_unit_price: Optional[float] = None
    discrepancy_amount: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class FeedbackEntry:
    """One feedback record per (invoice, config) reviewed by the user."""

    feedback_id: str
    timestamp: str  # ISO 8601
    invoice_id: str
    config_name: str

    # Optional run linkage (the Excel file the feedback was given on)
    run_filename: Optional[str] = None

    # Snapshot from model output (so feedback is self-contained)
    supplier_extracted: Optional[str] = None
    invoice_number_extracted: Optional[str] = None

    # User-tagged metadata
    supplier_canonical: Optional[str] = None  # normalized brand, e.g. "STARK"
    agreement_type: str = "price_agreement"
    agreement_notes: Optional[str] = None

    # Per-line verdicts on what the model flagged
    line_feedbacks: list[LineFeedback] = field(default_factory=list)

    # Discrepancies the model missed entirely
    missed_discrepancies: list[MissedDiscrepancy] = field(default_factory=list)

    # Overall judgment
    overall_verdict: str = "acceptable"
    overall_notes: Optional[str] = None

    @classmethod
    def new(cls, invoice_id: str, config_name: str, **kwargs) -> "FeedbackEntry":
        return cls(
            feedback_id=uuid.uuid4().hex,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            invoice_id=invoice_id,
            config_name=config_name,
            **kwargs,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackEntry":
        line_feedbacks = [
            LineFeedback(**lf) for lf in d.pop("line_feedbacks", []) or []
        ]
        missed = [
            MissedDiscrepancy(**md)
            for md in d.pop("missed_discrepancies", []) or []
        ]
        return cls(line_feedbacks=line_feedbacks, missed_discrepancies=missed, **d)


class FeedbackStore:
    """JSONL-backed feedback store, thread-safe."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read_all(self) -> list[FeedbackEntry]:
        if not self.path.exists():
            return []
        out: list[FeedbackEntry] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(FeedbackEntry.from_dict(json.loads(line)))
                except Exception:
                    # Skip corrupt lines but keep going. Better to lose one
                    # bad record than to refuse to load the rest.
                    continue
        return out

    def list_all(self) -> list[FeedbackEntry]:
        with self._lock:
            return self._read_all()

    def get(self, feedback_id: str) -> Optional[FeedbackEntry]:
        for e in self.list_all():
            if e.feedback_id == feedback_id:
                return e
        return None

    def find(self, invoice_id: str, config_name: str) -> Optional[FeedbackEntry]:
        """Latest feedback for this (invoice, config) pair, if any."""
        candidates = [
            e
            for e in self.list_all()
            if e.invoice_id == invoice_id and e.config_name == config_name
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.timestamp)

    def save(self, entry: FeedbackEntry) -> None:
        """Append or replace by feedback_id."""
        with self._lock:
            existing = self._read_all()
            replaced = False
            for i, e in enumerate(existing):
                if e.feedback_id == entry.feedback_id:
                    existing[i] = entry
                    replaced = True
                    break
            if not replaced:
                existing.append(entry)
            self._write_all(existing)

    def delete(self, feedback_id: str) -> bool:
        with self._lock:
            existing = self._read_all()
            new = [e for e in existing if e.feedback_id != feedback_id]
            if len(new) == len(existing):
                return False
            self._write_all(new)
            return True

    def _write_all(self, entries: list[FeedbackEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to tmp then rename so we never leave a half-written file.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False, default=str))
                f.write("\n")
        tmp.replace(self.path)


def build_entry_from_pred(
    invoice_id: str,
    config_name: str,
    pred: dict,
    *,
    run_filename: Optional[str] = None,
) -> FeedbackEntry:
    """Pre-fill a FeedbackEntry skeleton from a model prediction.

    The line_feedbacks list gets one slot per discrepant line item with
    the model's claims pre-populated. The user just sets the verdict
    and any corrections. missed_discrepancies starts empty; the user
    appends rows for things the model didn't catch.
    """
    line_feedbacks: list[LineFeedback] = []
    for i, li in enumerate(pred.get("line_items") or []):
        if not li.get("has_discrepancy"):
            continue
        line_feedbacks.append(
            LineFeedback(
                line_index=i,
                model_item_number=(
                    str(li.get("item_number")).strip()
                    if li.get("item_number") is not None
                    else None
                ),
                model_description=(li.get("description") or "").strip() or None,
                model_quantity=_to_float(li.get("quantity")),
                model_unit_price=_to_float(li.get("unit_price")),
                model_agreed_unit_price=_to_float(li.get("agreed_unit_price")),
                model_has_discrepancy=bool(li.get("has_discrepancy")),
                model_discrepancy_amount=_to_float(li.get("discrepancy_amount")),
            )
        )
    return FeedbackEntry.new(
        invoice_id=invoice_id,
        config_name=config_name,
        run_filename=run_filename,
        supplier_extracted=pred.get("supplier_name"),
        invoice_number_extracted=pred.get("invoice_number"),
        line_feedbacks=line_feedbacks,
    )


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

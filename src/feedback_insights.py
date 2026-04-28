"""Aggregations over collected feedback.

Computes per-config / per-supplier / per-agreement-type metrics so the
user can see at a glance: which model is best on STARK invoices, which
agreement type is hardest, what kind of discrepancies get missed most
often. The output of this module drives the /feedback/insights page
and informs prompt-tuning decisions.

All metrics are derived directly from FeedbackEntry records — no API
calls, no model inference, just counting.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .feedback import FeedbackEntry, LineFeedback, MissedDiscrepancy


@dataclass
class ConfigMetrics:
    """Per-config performance summary derived from human feedback."""

    config_name: str
    feedback_count: int  # how many (invoice, config) pairs reviewed
    flagged_total: int   # total discrepancies the model claimed
    correct: int         # of those, marked "correct"
    false_positive: int  # marked "false_positive"
    wrong_amount: int    # marked "wrong_amount" (right idea, wrong number)
    no_match_needed: int
    missed_count: int    # total discrepancies the user added as missed

    @property
    def precision(self) -> float | None:
        """Of all model claims, fraction that were correct."""
        if self.flagged_total == 0:
            return None
        return self.correct / self.flagged_total

    @property
    def recall_estimate(self) -> float | None:
        """Of all true discrepancies (correct + missed), fraction model caught.

        wrong_amount counts as half-caught: model identified the line but the
        amount was off. False positives don't count toward recall.
        """
        true_pos = self.correct + 0.5 * self.wrong_amount
        true_total = true_pos + self.missed_count
        if true_total == 0:
            return None
        return true_pos / true_total


@dataclass
class GroupMetrics:
    """Aggregated metrics for a slice (e.g. per-supplier, per-agreement-type)."""

    label: str
    feedback_count: int
    flagged_total: int
    correct: int
    false_positive: int
    wrong_amount: int
    missed_count: int

    @property
    def precision(self) -> float | None:
        if self.flagged_total == 0:
            return None
        return self.correct / self.flagged_total

    @property
    def recall_estimate(self) -> float | None:
        true_pos = self.correct + 0.5 * self.wrong_amount
        true_total = true_pos + self.missed_count
        if true_total == 0:
            return None
        return true_pos / true_total


@dataclass
class MissPattern:
    """A common pattern in missed discrepancies."""

    name: str
    count: int
    examples: list[str]


def per_config(entries: Iterable[FeedbackEntry]) -> list[ConfigMetrics]:
    by_cfg: dict[str, dict] = defaultdict(
        lambda: {"feedback": 0, "correct": 0, "fp": 0, "wrong": 0, "no_match": 0, "missed": 0}
    )
    for e in entries:
        d = by_cfg[e.config_name]
        d["feedback"] += 1
        for lf in e.line_feedbacks:
            if lf.verdict == "correct":
                d["correct"] += 1
            elif lf.verdict == "false_positive":
                d["fp"] += 1
            elif lf.verdict == "wrong_amount":
                d["wrong"] += 1
            elif lf.verdict == "no_match_needed":
                d["no_match"] += 1
        d["missed"] += len(e.missed_discrepancies)
    out: list[ConfigMetrics] = []
    for cfg, d in by_cfg.items():
        flagged_total = d["correct"] + d["fp"] + d["wrong"] + d["no_match"]
        out.append(
            ConfigMetrics(
                config_name=cfg,
                feedback_count=d["feedback"],
                flagged_total=flagged_total,
                correct=d["correct"],
                false_positive=d["fp"],
                wrong_amount=d["wrong"],
                no_match_needed=d["no_match"],
                missed_count=d["missed"],
            )
        )
    out.sort(key=lambda m: -(m.precision or 0))
    return out


def per_supplier(entries: Iterable[FeedbackEntry]) -> list[GroupMetrics]:
    return _group_by(entries, lambda e: (e.supplier_canonical or e.supplier_extracted or "(unknown)"))


def per_agreement_type(entries: Iterable[FeedbackEntry]) -> list[GroupMetrics]:
    return _group_by(entries, lambda e: e.agreement_type or "other")


def _group_by(entries: Iterable[FeedbackEntry], key) -> list[GroupMetrics]:
    by: dict[str, dict] = defaultdict(
        lambda: {"feedback": 0, "correct": 0, "fp": 0, "wrong": 0, "missed": 0}
    )
    for e in entries:
        k = key(e)
        d = by[k]
        d["feedback"] += 1
        for lf in e.line_feedbacks:
            if lf.verdict == "correct":
                d["correct"] += 1
            elif lf.verdict == "false_positive":
                d["fp"] += 1
            elif lf.verdict == "wrong_amount":
                d["wrong"] += 1
        d["missed"] += len(e.missed_discrepancies)
    out: list[GroupMetrics] = []
    for label, d in by.items():
        flagged_total = d["correct"] + d["fp"] + d["wrong"]
        out.append(
            GroupMetrics(
                label=label,
                feedback_count=d["feedback"],
                flagged_total=flagged_total,
                correct=d["correct"],
                false_positive=d["fp"],
                wrong_amount=d["wrong"],
                missed_count=d["missed"],
            )
        )
    out.sort(key=lambda m: -m.feedback_count)
    return out


def find_miss_patterns(entries: Iterable[FeedbackEntry]) -> list[MissPattern]:
    """Heuristic patterns in missed discrepancies.

    These are deliberately simple — they help the user see at a glance
    where prompts are weakest. More sophisticated analysis can be done
    by exporting the raw feedback to Excel.
    """
    small_per_unit = []  # per-unit diff < 1 kr
    no_item_number = []
    large_qty_small_per_unit = []  # qty > 5 AND per-unit diff < 1 kr
    medium_per_unit = []  # 1-5 kr per-unit diff (often missed too)
    rebate_missing = []  # heuristic: notes mention "rabat" / "bonus"

    for e in entries:
        for md in e.missed_discrepancies:
            label = f"{e.config_name} :: {e.invoice_id} :: {md.description[:40]}"
            qty = md.quantity or 0
            unit = md.unit_price
            agreed = md.correct_agreed_unit_price
            per_unit_diff = (
                abs(unit - agreed)
                if (unit is not None and agreed is not None)
                else None
            )
            if per_unit_diff is not None and per_unit_diff < 1.0:
                small_per_unit.append(label)
                if qty > 5:
                    large_qty_small_per_unit.append(label)
            elif per_unit_diff is not None and 1.0 <= per_unit_diff <= 5.0:
                medium_per_unit.append(label)
            if not (md.item_number or "").strip():
                no_item_number.append(label)
            notes = (md.notes or "").lower()
            if "rabat" in notes or "bonus" in notes or "afslag" in notes:
                rebate_missing.append(label)

    patterns: list[MissPattern] = []
    if small_per_unit:
        patterns.append(MissPattern(
            name="Small per-unit diff (<1 kr)",
            count=len(small_per_unit),
            examples=small_per_unit[:5],
        ))
    if large_qty_small_per_unit:
        patterns.append(MissPattern(
            name="Large quantity × small per-unit diff",
            count=len(large_qty_small_per_unit),
            examples=large_qty_small_per_unit[:5],
        ))
    if medium_per_unit:
        patterns.append(MissPattern(
            name="Medium per-unit diff (1-5 kr)",
            count=len(medium_per_unit),
            examples=medium_per_unit[:5],
        ))
    if no_item_number:
        patterns.append(MissPattern(
            name="Missing item_number on the missed line",
            count=len(no_item_number),
            examples=no_item_number[:5],
        ))
    if rebate_missing:
        patterns.append(MissPattern(
            name="Rebate / bonus context",
            count=len(rebate_missing),
            examples=rebate_missing[:5],
        ))
    return patterns


@dataclass
class OverallSummary:
    total_entries: int
    total_invoices: int
    total_configs: int
    total_correct: int
    total_false_positive: int
    total_wrong_amount: int
    total_missed: int


def overall(entries: Iterable[FeedbackEntry]) -> OverallSummary:
    entries = list(entries)
    invoices = {e.invoice_id for e in entries}
    configs = {e.config_name for e in entries}
    correct = sum(1 for e in entries for lf in e.line_feedbacks if lf.verdict == "correct")
    fp = sum(1 for e in entries for lf in e.line_feedbacks if lf.verdict == "false_positive")
    wrong = sum(1 for e in entries for lf in e.line_feedbacks if lf.verdict == "wrong_amount")
    missed = sum(len(e.missed_discrepancies) for e in entries)
    return OverallSummary(
        total_entries=len(entries),
        total_invoices=len(invoices),
        total_configs=len(configs),
        total_correct=correct,
        total_false_positive=fp,
        total_wrong_amount=wrong,
        total_missed=missed,
    )

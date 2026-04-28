"""Regression testing against captured feedback.

Once the user has feedback for a few invoices, we can use it as a
labeled golden set. When they tweak prompts, run the same invoices
through the new prompts and compare against the old feedback labels:

- Did errors that were correctly flagged before still get flagged?
- Did errors that were missed before now get caught? (Improvement)
- Did errors that were correctly flagged before now get missed? (Regression)
- Did false positives go up or down?

The output is a per-invoice / per-config diff so the user can decide
whether the prompt change is a net win.

This module does NOT modify the runner. It calls runner.run() with
limited jobs and post-processes the results in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .feedback import FeedbackEntry, MissedDiscrepancy


# A small absolute tolerance for matching a "new" model finding against
# a known correct/missed entry from feedback. Same tolerance the prompt
# itself uses for discrepancy detection.
NUMERIC_TOL = 0.02


@dataclass
class LineMatch:
    """How a model line matches up against feedback ground truth."""

    feedback_line_index: Optional[int]  # which line_feedback / missed entry it matched
    matched_via: str  # "item_number" | "description" | "none"
    new_pred: dict
    expected_verdict: str  # what feedback said
    actual_outcome: str    # "matches_correct" | "matches_missed" | "matches_fp" | "new_finding"


@dataclass
class InvoiceRegressionResult:
    invoice_id: str
    config_name: str

    # Counts
    previously_correct_still_caught: int = 0
    previously_correct_now_missed: int = 0  # REGRESSION
    previously_missed_now_caught: int = 0   # IMPROVEMENT
    previously_missed_still_missed: int = 0
    previously_fp_now_dropped: int = 0      # IMPROVEMENT
    previously_fp_still_present: int = 0
    new_findings: int = 0                    # findings not in any feedback bucket

    notes: list[str] = field(default_factory=list)


@dataclass
class RegressionReport:
    feedback_used: int
    invoices: list[InvoiceRegressionResult] = field(default_factory=list)

    def overall_improvement_count(self) -> int:
        return sum(
            r.previously_missed_now_caught + r.previously_fp_now_dropped
            for r in self.invoices
        )

    def overall_regression_count(self) -> int:
        return sum(r.previously_correct_now_missed for r in self.invoices)

    def overall_stable_count(self) -> int:
        return sum(
            r.previously_correct_still_caught
            + r.previously_missed_still_missed
            + r.previously_fp_still_present
            for r in self.invoices
        )


def _norm_item_number(s: Optional[str]) -> Optional[str]:
    """Normalize item_number for matching: strip whitespace and separators."""
    if s is None:
        return None
    cleaned = "".join(ch for ch in str(s) if ch.isalnum())
    return cleaned.lower() or None


def _line_signature(line: dict) -> tuple[Optional[str], str]:
    item = _norm_item_number(line.get("item_number"))
    desc = (line.get("description") or "").strip().lower()
    return (item, desc[:60])


def _matches(a_sig: tuple, b_sig: tuple) -> bool:
    """Two lines refer to the same item if item_numbers match (when both
    available) or descriptions overlap on the prefix."""
    if a_sig[0] and b_sig[0]:
        return a_sig[0] == b_sig[0]
    if not a_sig[1] or not b_sig[1]:
        return False
    a, b = a_sig[1], b_sig[1]
    # Substring match either way; useful for descriptions like
    # "RAW STANDARD MDF PLADE 16 X 1220" vs shorter "MDF PLADE 16".
    return a.startswith(b) or b.startswith(a) or (a in b) or (b in a)


def compare(
    pred: dict,
    feedback: FeedbackEntry,
) -> InvoiceRegressionResult:
    """Compare a fresh model prediction against a single feedback entry."""
    result = InvoiceRegressionResult(
        invoice_id=feedback.invoice_id,
        config_name=feedback.config_name,
    )

    # New findings the model flagged this run
    new_lines = [
        li for li in (pred.get("line_items") or []) if li.get("has_discrepancy")
    ]
    new_sigs = [_line_signature(li) for li in new_lines]
    new_used = [False] * len(new_lines)

    # 1) Per previously-flagged line in the original run, did the new prediction
    #    keep it (correct still caught) or drop it (regression)?
    for lf in feedback.line_feedbacks:
        # Build the "feedback view" of that line
        fb_line = {
            "item_number": lf.model_item_number,
            "description": lf.model_description,
        }
        fb_sig = _line_signature(fb_line)
        match_idx = None
        for i, sig in enumerate(new_sigs):
            if new_used[i]:
                continue
            if _matches(fb_sig, sig):
                match_idx = i
                break
        if match_idx is not None:
            new_used[match_idx] = True
            if lf.verdict == "correct":
                result.previously_correct_still_caught += 1
            elif lf.verdict == "false_positive":
                result.previously_fp_still_present += 1
            elif lf.verdict == "wrong_amount":
                # Model still flags this line; whether amount is now right
                # is interesting but beyond this MVP — count as still-caught.
                result.previously_correct_still_caught += 1
            elif lf.verdict == "no_match_needed":
                # Treat as benign noise either way.
                pass
        else:
            # New run did NOT flag this line.
            if lf.verdict == "correct":
                result.previously_correct_now_missed += 1
                result.notes.append(
                    f"REGRESSION: line '{(lf.model_description or '')[:40]}' "
                    f"({lf.model_item_number or 'no varenr'}) was correctly flagged before, missed now."
                )
            elif lf.verdict == "false_positive":
                result.previously_fp_now_dropped += 1
            elif lf.verdict == "wrong_amount":
                # Was "right idea, wrong amount"; now not flagged at all → regression
                result.previously_correct_now_missed += 1

    # 2) Per missed discrepancy from feedback: did the new run catch it?
    for md in feedback.missed_discrepancies:
        md_sig = _line_signature({
            "item_number": md.item_number,
            "description": md.description,
        })
        match_idx = None
        for i, sig in enumerate(new_sigs):
            if new_used[i]:
                continue
            if _matches(md_sig, sig):
                match_idx = i
                break
        if match_idx is not None:
            new_used[match_idx] = True
            result.previously_missed_now_caught += 1
            result.notes.append(
                f"IMPROVEMENT: line '{md.description[:40]}' was missed before, caught now."
            )
        else:
            result.previously_missed_still_missed += 1

    # 3) Whatever new findings remain weren't in feedback at all
    for i, used in enumerate(new_used):
        if not used:
            result.new_findings += 1

    return result


def jobs_for_regression(
    feedback_entries: list[FeedbackEntry],
    invoices_dir: Path,
    agreements_dir: Path,
) -> tuple[list, dict[tuple[str, str], FeedbackEntry]]:
    """Build (jobs, feedback_index) for re-running through the runner.

    Returns:
        - list of InvoiceJob (compatible with src.runner.run_one)
        - mapping {(invoice_id, config_name) -> FeedbackEntry} for comparison
    """
    from .runner import InvoiceJob

    # Unique invoice_ids across feedback
    invoice_ids = {e.invoice_id for e in feedback_entries}

    jobs: list = []
    for inv_id in invoice_ids:
        inv_path = invoices_dir / inv_id
        if not inv_path.exists():
            # Skip invoices that aren't on disk anymore
            continue
        # Use all available agreements as context (mirrors no-GT path in runner)
        ag_paths = []
        if agreements_dir.exists():
            for ext in (".pdf", ".xlsx"):
                ag_paths.extend(sorted(agreements_dir.glob(f"*{ext}")))
        jobs.append(
            InvoiceJob(
                invoice_id=inv_id,
                invoice_path=inv_path,
                agreement_paths=list(ag_paths),
                ground_truth=None,
            )
        )

    feedback_index: dict[tuple[str, str], FeedbackEntry] = {
        (e.invoice_id, e.config_name): e for e in feedback_entries
    }
    return jobs, feedback_index

"""LLM-driven prompt-improvement assistant.

The user has accumulated feedback. This module:
  1) Composes a structured meta-prompt that tells Claude exactly how to
     analyze the feedback (with hard rules to prevent the AI from "building
     on wrong premises").
  2) Calls Claude with the current prompts + feedback summary + insights.
  3) Validates the response — suggestions must cite >=3 feedback IDs that
     actually exist, the `find` text must appear verbatim in the named
     prompt block, etc. Hallucinated suggestions are filtered out with a
     visible reason so the user can see what was rejected and why.

Suggestions are NEVER auto-applied. The webapp shows them as cards; the
user clicks Apply / Reject / Tweak per suggestion.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .feedback import FeedbackEntry, FeedbackStore
from .feedback_insights import (
    find_miss_patterns,
    overall,
    per_agreement_type,
    per_config,
    per_supplier,
)
from .prompts import PromptTemplates


META_PROMPT_HEADER = """You are a prompt-engineering assistant for an LLM-based Danish invoice
auditor. Real users have reviewed the auditor's output and given feedback,
including marking errors the auditor missed and false positives it raised.
Your job is to analyze the feedback and propose specific, conservative
edits to the audit prompts to improve precision and recall.

CRITICAL RULES — violating any of these means the suggestion will be
filtered out before reaching the user:

1. EVERY suggestion MUST cite at least 3 specific feedback entries by their
   feedback_id. If a pattern is seen in fewer than 3 cases, do NOT suggest
   a change for it. Note it in considered_but_rejected instead.

2. EVERY suggestion MUST be a concrete find/replace edit. Specify:
   - Which prompt block: "intro" | "instructions_extraction_only" |
     "instructions_with_agreement" | "reasoning_instructions"
   - The exact text to find (must appear verbatim in current prompt — no
     paraphrasing, no fuzzy matching)
   - The exact text to replace it with
   No "general improvements" or "rewrite section X" — text-level diffs only.

3. Confidence calibration:
   - "high": 5+ supporting cases, clear pattern, low overfitting risk
   - "medium": 3-4 cases, plausible pattern, generalization needed
   - "low": 3 borderline cases — only suggest if the change is clearly safe

4. You MUST output "considered_but_rejected" — patterns you noticed but
   chose NOT to act on, with reasons. This transparency is required.

5. Be conservative. Prefer adding a sentence over rewriting a section.
   Prefer reinforcing an existing rule over introducing a new one.

6. Never base a suggestion on a single supplier's quirks unless multiple
   feedback entries from that supplier confirm the pattern AND the change
   is phrased generically (no hardcoding "STARK" etc.) so it generalizes.

7. Output STRICTLY valid JSON. No prose, no markdown, no code fences.
"""


META_PROMPT_OUTPUT_SPEC = """OUTPUT FORMAT (strictly valid JSON, no prose, no code fences):

{
  "overall_diagnosis": "<2-4 sentences: what is working, what is not>",
  "suggestions": [
    {
      "block": "instructions_with_agreement",
      "find": "<exact substring of the current block, copy verbatim>",
      "replace": "<replacement text>",
      "rationale": "<1-3 sentences explaining why>",
      "confidence": "high",
      "supporting_feedback_ids": ["abc123...", "def456...", "ghi789..."],
      "evidence_summary": "<what these cases have in common, 1 sentence>"
    }
  ],
  "considered_but_rejected": [
    {
      "pattern": "<observed pattern>",
      "why_rejected": "<why no change was suggested>"
    }
  ]
}
"""


@dataclass
class Suggestion:
    """One concrete prompt edit suggested by the LLM."""

    suggestion_id: str
    block: str  # intro | instructions_extraction_only | instructions_with_agreement | reasoning_instructions
    find: str
    replace: str
    rationale: str
    confidence: str  # high | medium | low
    supporting_feedback_ids: list[str]
    evidence_summary: str = ""

    # Set after validation:
    is_valid: bool = True
    rejection_reason: Optional[str] = None  # populated when validation fails


@dataclass
class RejectedConsideration:
    pattern: str
    why_rejected: str


@dataclass
class AutoTuneResult:
    """Full output of one autotune run."""

    run_id: str
    timestamp: str
    overall_diagnosis: str
    suggestions: list[Suggestion] = field(default_factory=list)
    rejected_by_llm: list[RejectedConsideration] = field(default_factory=list)
    rejected_by_validation: list[Suggestion] = field(default_factory=list)

    # Snapshot of input the LLM saw (for transparency)
    feedback_count: int = 0
    feedback_id_list: list[str] = field(default_factory=list)
    raw_llm_response: str = ""
    meta_prompt_excerpt: str = ""  # first 1000 chars of meta-prompt for debug
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _summarize_feedback_for_llm(entries: list[FeedbackEntry]) -> str:
    """Compact text summary of feedback entries for the LLM context.

    Keeps the JSON small enough to fit in 4-8k input tokens even with 50+
    entries. Includes feedback_id so the LLM can cite specific cases.
    """
    out: list[str] = []
    for e in entries:
        correct = sum(1 for lf in e.line_feedbacks if lf.verdict == "correct")
        fp = sum(1 for lf in e.line_feedbacks if lf.verdict == "false_positive")
        wrong = sum(1 for lf in e.line_feedbacks if lf.verdict == "wrong_amount")
        miss = len(e.missed_discrepancies)

        head = (
            f"feedback_id={e.feedback_id[:8]} model={e.config_name} "
            f"supplier={e.supplier_canonical or e.supplier_extracted or '?'} "
            f"agr_type={e.agreement_type} verdict={e.overall_verdict} "
            f"correct={correct} fp={fp} wrong={wrong} missed={miss}"
        )
        details: list[str] = []
        for lf in e.line_feedbacks:
            if lf.verdict in ("false_positive", "wrong_amount"):
                details.append(
                    f"  [{lf.verdict}] item={lf.model_item_number or '-'} "
                    f"desc={(lf.model_description or '')[:60]} "
                    f"model_says: faktureret={lf.model_unit_price} aftalt={lf.model_agreed_unit_price} "
                    f"diff={lf.model_discrepancy_amount}"
                    + (f" notes={lf.notes[:80]}" if lf.notes else "")
                )
        for md in e.missed_discrepancies:
            details.append(
                f"  [missed] item={md.item_number or '-'} desc={md.description[:60]} "
                f"qty={md.quantity} faktureret={md.unit_price} aftalt={md.correct_agreed_unit_price} "
                f"diff={md.discrepancy_amount}"
                + (f" notes={md.notes[:80]}" if md.notes else "")
            )
        if e.agreement_notes:
            details.append(f"  agr_notes: {e.agreement_notes[:120]}")
        if e.overall_notes:
            details.append(f"  overall_notes: {e.overall_notes[:120]}")
        block = head + ("\n" + "\n".join(details) if details else "")
        out.append(block)
    return "\n\n".join(out)


def _summarize_insights(entries: list[FeedbackEntry]) -> str:
    summary = overall(entries)
    by_cfg = per_config(entries)
    by_sup = per_supplier(entries)
    by_type = per_agreement_type(entries)
    patterns = find_miss_patterns(entries)

    lines: list[str] = []
    lines.append(
        f"Overall: {summary.total_entries} reviews · {summary.total_correct} correct · "
        f"{summary.total_false_positive} fp · {summary.total_wrong_amount} wrong_amount · "
        f"{summary.total_missed} missed"
    )
    lines.append("\nPer model:")
    for m in by_cfg:
        prec = f"{m.precision*100:.0f}%" if m.precision is not None else "—"
        rec = f"{m.recall_estimate*100:.0f}%" if m.recall_estimate is not None else "—"
        lines.append(
            f"  {m.config_name}: precision={prec} recall={rec} "
            f"(correct={m.correct} fp={m.false_positive} wrong={m.wrong_amount} missed={m.missed_count})"
        )
    if by_sup:
        lines.append("\nPer supplier:")
        for g in by_sup:
            prec = f"{g.precision*100:.0f}%" if g.precision is not None else "—"
            rec = f"{g.recall_estimate*100:.0f}%" if g.recall_estimate is not None else "—"
            lines.append(f"  {g.label}: precision={prec} recall={rec} (n={g.feedback_count})")
    if by_type:
        lines.append("\nPer agreement type:")
        for g in by_type:
            prec = f"{g.precision*100:.0f}%" if g.precision is not None else "—"
            rec = f"{g.recall_estimate*100:.0f}%" if g.recall_estimate is not None else "—"
            lines.append(f"  {g.label}: precision={prec} recall={rec} (n={g.feedback_count})")
    if patterns:
        lines.append("\nMiss patterns (heuristic):")
        for p in patterns:
            lines.append(f"  {p.name}: {p.count} cases")
    return "\n".join(lines)


def _format_current_prompts(t: PromptTemplates) -> str:
    return (
        f"=== BLOCK: intro ===\n{t.intro}\n\n"
        f"=== BLOCK: instructions_extraction_only ===\n{t.instructions_extraction_only}\n\n"
        f"=== BLOCK: instructions_with_agreement ===\n{t.instructions_with_agreement}\n\n"
        f"=== BLOCK: reasoning_instructions ===\n{t.reasoning_instructions}\n"
    )


def build_meta_prompt(
    templates: PromptTemplates, entries: list[FeedbackEntry]
) -> str:
    parts = [
        META_PROMPT_HEADER,
        "\n--- CURRENT PROMPTS ---\n",
        _format_current_prompts(templates),
        "\n--- AGGREGATED METRICS ---\n",
        _summarize_insights(entries),
        "\n--- FEEDBACK ENTRIES ---\n",
        _summarize_feedback_for_llm(entries),
        "\n",
        META_PROMPT_OUTPUT_SPEC,
    ]
    return "\n".join(parts)


def _validate_suggestion(
    s: Suggestion,
    templates: PromptTemplates,
    valid_feedback_ids: set[str],
) -> Suggestion:
    """Apply the hard rules. Sets is_valid=False with a rejection_reason if
    the suggestion fails any check.
    """
    valid_blocks = (
        "intro",
        "instructions_extraction_only",
        "instructions_with_agreement",
        "reasoning_instructions",
    )
    if s.block not in valid_blocks:
        s.is_valid = False
        s.rejection_reason = f"Unknown block '{s.block}'"
        return s

    block_text = getattr(templates, s.block, "")
    if not s.find:
        s.is_valid = False
        s.rejection_reason = "Empty find string"
        return s
    if s.find not in block_text:
        # Try a forgiving prefix match for whitespace-only differences.
        normalized_find = re.sub(r"\s+", " ", s.find).strip()
        normalized_block = re.sub(r"\s+", " ", block_text).strip()
        if normalized_find not in normalized_block:
            s.is_valid = False
            s.rejection_reason = (
                "find-text not found verbatim in current prompt block"
            )
            return s

    # Match feedback IDs to actual entries (allow short prefix matches)
    matched_ids: list[str] = []
    for cited in s.supporting_feedback_ids or []:
        cited_str = str(cited).strip()
        if cited_str in valid_feedback_ids:
            matched_ids.append(cited_str)
            continue
        for fid in valid_feedback_ids:
            if fid.startswith(cited_str) and len(cited_str) >= 6:
                matched_ids.append(fid)
                break
    s.supporting_feedback_ids = matched_ids
    if len(matched_ids) < 3:
        s.is_valid = False
        s.rejection_reason = (
            f"Only {len(matched_ids)} of cited feedback IDs match real entries; "
            "minimum is 3"
        )
        return s

    if s.confidence not in ("high", "medium", "low"):
        s.confidence = "low"

    return s


def _parse_llm_response(
    raw: str,
    templates: PromptTemplates,
    valid_feedback_ids: set[str],
) -> tuple[str, list[Suggestion], list[Suggestion], list[RejectedConsideration]]:
    """Returns (overall_diagnosis, valid_suggestions, invalid_suggestions, rejected_by_llm)."""
    # Strip code fences if model added them
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("LLM response was not valid JSON")
        obj = json.loads(m.group(0))

    diagnosis = (obj.get("overall_diagnosis") or "").strip()
    raw_suggestions = obj.get("suggestions") or []
    raw_rejected = obj.get("considered_but_rejected") or []

    valid: list[Suggestion] = []
    invalid: list[Suggestion] = []
    for raw_s in raw_suggestions:
        if not isinstance(raw_s, dict):
            continue
        s = Suggestion(
            suggestion_id=uuid.uuid4().hex[:8],
            block=str(raw_s.get("block") or "").strip(),
            find=str(raw_s.get("find") or ""),
            replace=str(raw_s.get("replace") or ""),
            rationale=str(raw_s.get("rationale") or "").strip(),
            confidence=str(raw_s.get("confidence") or "low").strip().lower(),
            supporting_feedback_ids=[
                str(x) for x in (raw_s.get("supporting_feedback_ids") or [])
            ],
            evidence_summary=str(raw_s.get("evidence_summary") or "").strip(),
        )
        s = _validate_suggestion(s, templates, valid_feedback_ids)
        if s.is_valid:
            valid.append(s)
        else:
            invalid.append(s)

    rejected = []
    for r in raw_rejected:
        if not isinstance(r, dict):
            continue
        rejected.append(
            RejectedConsideration(
                pattern=str(r.get("pattern") or "").strip(),
                why_rejected=str(r.get("why_rejected") or "").strip(),
            )
        )

    return diagnosis, valid, invalid, rejected


def analyze_with_llm(
    templates: PromptTemplates,
    entries: list[FeedbackEntry],
    *,
    claude_client,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> AutoTuneResult:
    """Run the LLM analysis. Returns a fully-populated AutoTuneResult."""
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().isoformat(timespec="seconds")
    valid_ids = {e.feedback_id for e in entries}

    if not entries:
        return AutoTuneResult(
            run_id=run_id, timestamp=ts,
            overall_diagnosis="No feedback to analyze. Capture some reviews first.",
            error="empty_feedback",
        )

    meta = build_meta_prompt(templates, entries)
    try:
        call = claude_client.call(
            model, meta, pdf_documents=[], cached_prefix_pdfs=[]
        )
    except Exception as e:
        return AutoTuneResult(
            run_id=run_id, timestamp=ts,
            overall_diagnosis="LLM call failed.",
            error=f"{type(e).__name__}: {e}",
            feedback_count=len(entries),
            feedback_id_list=sorted(valid_ids),
            meta_prompt_excerpt=meta[:1000],
        )

    if call.error:
        return AutoTuneResult(
            run_id=run_id, timestamp=ts,
            overall_diagnosis="LLM returned an error.",
            error=call.error,
            feedback_count=len(entries),
            feedback_id_list=sorted(valid_ids),
            meta_prompt_excerpt=meta[:1000],
            raw_llm_response=call.raw_text or "",
        )

    try:
        diagnosis, valid, invalid, rejected = _parse_llm_response(
            call.raw_text, templates, valid_ids
        )
    except Exception as e:
        return AutoTuneResult(
            run_id=run_id, timestamp=ts,
            overall_diagnosis="LLM response failed validation.",
            error=f"parse: {e}",
            feedback_count=len(entries),
            feedback_id_list=sorted(valid_ids),
            meta_prompt_excerpt=meta[:1000],
            raw_llm_response=call.raw_text or "",
        )

    return AutoTuneResult(
        run_id=run_id, timestamp=ts,
        overall_diagnosis=diagnosis,
        suggestions=valid,
        rejected_by_validation=invalid,
        rejected_by_llm=rejected,
        feedback_count=len(entries),
        feedback_id_list=sorted(valid_ids),
        raw_llm_response=call.raw_text or "",
        meta_prompt_excerpt=meta[:1000],
    )


def apply_suggestion(
    templates: PromptTemplates, suggestion: Suggestion
) -> PromptTemplates:
    """Return a NEW PromptTemplates with the suggestion applied. Leaves
    the original unchanged so the caller controls when to persist."""
    if not suggestion.is_valid:
        raise ValueError(
            f"Cannot apply invalid suggestion: {suggestion.rejection_reason}"
        )
    block_text = getattr(templates, suggestion.block, "")
    if suggestion.find in block_text:
        new_text = block_text.replace(suggestion.find, suggestion.replace, 1)
    else:
        # Fall back to whitespace-normalized search and replace
        normalized_find = re.sub(r"\s+", " ", suggestion.find).strip()
        # Find a match in block_text
        normalized_block = re.sub(r"\s+", " ", block_text)
        if normalized_find not in normalized_block:
            raise ValueError(
                "find-text not found in current block (it may have been "
                "edited since the suggestion was generated)"
            )
        # Conservative fallback: replace at the position of the normalized
        # match. We do this by reconstructing — find the original substring
        # whose normalized form equals normalized_find.
        # Simple heuristic: collapse multiple whitespace runs to a single
        # space in the block, find the match, then re-replace in the
        # original. This is imperfect for edge cases but safer than failing.
        new_text = re.sub(
            r"\s+".join(re.escape(w) for w in suggestion.find.split()),
            lambda _m: suggestion.replace,
            block_text,
            count=1,
        )
        if new_text == block_text:
            raise ValueError("find-text could not be located after normalization")

    new_templates = PromptTemplates(
        intro=templates.intro,
        instructions_extraction_only=templates.instructions_extraction_only,
        instructions_with_agreement=templates.instructions_with_agreement,
        reasoning_instructions=templates.reasoning_instructions,
    )
    setattr(new_templates, suggestion.block, new_text)
    return new_templates


# In-memory storage of autotune run results (single-process, like the main RUNS)
class AutoTuneStore:
    """Persists autotune results to disk so a result page survives a worker restart."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def save(self, result: AutoTuneResult) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[result.run_id] = result.to_dict()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def get(self, run_id: str) -> Optional[AutoTuneResult]:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        d = data.get(run_id)
        if not d:
            return None
        return _result_from_dict(d)

    def list_recent(self, limit: int = 20) -> list[AutoTuneResult]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        results = [_result_from_dict(d) for d in data.values()]
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]


def _result_from_dict(d: dict) -> AutoTuneResult:
    suggestions = [Suggestion(**s) for s in d.get("suggestions", [])]
    rejected_validation = [Suggestion(**s) for s in d.get("rejected_by_validation", [])]
    rejected_llm = [
        RejectedConsideration(**r) for r in d.get("rejected_by_llm", [])
    ]
    return AutoTuneResult(
        run_id=d["run_id"],
        timestamp=d["timestamp"],
        overall_diagnosis=d.get("overall_diagnosis", ""),
        suggestions=suggestions,
        rejected_by_validation=rejected_validation,
        rejected_by_llm=rejected_llm,
        feedback_count=d.get("feedback_count", 0),
        feedback_id_list=d.get("feedback_id_list", []),
        raw_llm_response=d.get("raw_llm_response", ""),
        meta_prompt_excerpt=d.get("meta_prompt_excerpt", ""),
        error=d.get("error"),
    )

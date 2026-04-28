"""Scoring: field extraction, price match, credit note, rebate, composite."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from rapidfuzz import fuzz


FIELD_LIST = [
    "supplier_name",
    "invoice_number",
    "invoice_date",
    "document_type",
    "currency",
    "subtotal",
    "vat",
    "total",
    "rebate_applied",
]


@dataclass
class FieldOutcome:
    field: str
    score: float  # 1.0 exact, 0.5 partial, 0.0 wrong/missing
    gt: Any = None
    pred: Any = None


@dataclass
class InvoiceScores:
    field_extraction_pct: float
    price_match_pct: float  # F1
    credit_note_pct: float  # 0 or 1
    rebate_pct: float
    composite: float
    per_field: list[FieldOutcome] = field(default_factory=list)
    price_match_details: dict = field(default_factory=dict)


# -------- normalization helpers --------

_DK_PREFIX_RE = re.compile(r"\b(a/s|aps|ivs|a\.m\.b\.a|k/s|p/s)\b", re.IGNORECASE)


def _norm_str(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _norm_supplier(s: Optional[str]) -> str:
    if s is None:
        return ""
    s2 = _DK_PREFIX_RE.sub("", str(s))
    return re.sub(r"\s+", " ", s2).strip().lower()


def _norm_date(s: Any) -> str:
    """Normalize to YYYY-MM-DD. Accepts YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY."""
    if s is None:
        return ""
    t = str(s).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t
    m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", t)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return t


def _norm_num(v: Any) -> Optional[float]:
    """Parse 1.234,56 and 1234.56 into float."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().replace(" ", "")
    if "," in t and "." in t:
        # Assume Danish: '.' thousands, ',' decimal.
        t = t.replace(".", "").replace(",", ".")
    elif "," in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _nums_match(a: Any, b: Any, tol: float = 0.01) -> bool:
    na, nb = _norm_num(a), _norm_num(b)
    if na is None or nb is None:
        return na == nb
    return abs(na - nb) <= tol


# -------- field extraction --------


def score_field_extraction(
    gt: dict, pred: dict, *, supplier_threshold: int = 90, numeric_tol: float = 0.01
) -> tuple[float, list[FieldOutcome]]:
    outcomes: list[FieldOutcome] = []

    for f in FIELD_LIST:
        if f not in gt:
            continue
        g = gt.get(f)
        p = pred.get(f) if pred else None

        if f == "supplier_name":
            gn, pn = _norm_supplier(g), _norm_supplier(p)
            if not gn and not pn:
                score = 1.0
            elif not pn:
                score = 0.0
            else:
                ratio = fuzz.token_sort_ratio(gn, pn)
                score = 1.0 if ratio >= supplier_threshold else 0.0
        elif f == "invoice_number":
            score = 1.0 if _norm_str(g) == _norm_str(p) else 0.0
        elif f == "invoice_date":
            score = 1.0 if _norm_date(g) == _norm_date(p) and _norm_date(g) else 0.0
        elif f in ("document_type", "currency"):
            score = 1.0 if _norm_str(g) == _norm_str(p) else 0.0
        else:  # numeric: subtotal, vat, total, rebate_applied
            score = 1.0 if _nums_match(g, p, numeric_tol) else 0.0

        outcomes.append(FieldOutcome(field=f, score=score, gt=g, pred=p))

    # Line-item scoring (each item contributes a sub-score; averaged as one "field").
    gt_items = gt.get("line_items") or []
    pred_items = pred.get("line_items") or [] if pred else []
    if gt_items or pred_items:
        li_score = _score_line_items(gt_items, pred_items)
        outcomes.append(
            FieldOutcome(
                field="line_items",
                score=li_score,
                gt=f"{len(gt_items)} items",
                pred=f"{len(pred_items)} items",
            )
        )

    if not outcomes:
        return 0.0, outcomes
    overall = sum(o.score for o in outcomes) / len(outcomes)
    return overall, outcomes


def _score_line_items(
    gt_items: list[dict], pred_items: list[dict], desc_threshold: int = 85
) -> float:
    if not gt_items:
        return 1.0 if not pred_items else 0.0
    matched_pred_idx: set[int] = set()
    total = 0.0
    for gt_item in gt_items:
        gd = _norm_str(gt_item.get("description"))
        best_i, best_r = -1, -1
        for i, pi in enumerate(pred_items):
            if i in matched_pred_idx:
                continue
            r = fuzz.token_sort_ratio(gd, _norm_str(pi.get("description")))
            if r > best_r:
                best_i, best_r = i, r
        if best_i == -1 or best_r < desc_threshold:
            continue
        matched_pred_idx.add(best_i)
        pi = pred_items[best_i]
        qty_ok = _nums_match(gt_item.get("quantity"), pi.get("quantity"))
        px_ok = _nums_match(gt_item.get("unit_price"), pi.get("unit_price"))
        if qty_ok and px_ok:
            total += 1.0
        else:
            total += 0.5
    return total / len(gt_items)


# -------- price match (discrepancy detection F1) --------


def score_price_match(gt: dict, pred: dict, desc_threshold: int = 85) -> tuple[float, dict]:
    """F1 over the binary 'has_discrepancy' label on matched line items."""
    gt_items = gt.get("line_items") or []
    pred_items = (pred.get("line_items") or []) if pred else []

    if not gt_items:
        # No line items in GT means no price-match signal.
        return 1.0, {"tp": 0, "fp": 0, "fn": 0, "note": "no line items in GT"}

    tp = fp = fn = 0
    used: set[int] = set()
    for gi in gt_items:
        gd = _norm_str(gi.get("description"))
        best_i, best_r = -1, -1
        for i, pi in enumerate(pred_items):
            if i in used:
                continue
            r = fuzz.token_sort_ratio(gd, _norm_str(pi.get("description")))
            if r > best_r:
                best_i, best_r = i, r
        gt_disc = bool(gi.get("has_discrepancy"))
        if best_i == -1 or best_r < desc_threshold:
            # Didn't find a match -> if GT had a discrepancy, that's a miss.
            if gt_disc:
                fn += 1
            continue
        used.add(best_i)
        pred_disc = bool(pred_items[best_i].get("has_discrepancy"))
        if gt_disc and pred_disc:
            tp += 1
        elif gt_disc and not pred_disc:
            fn += 1
        elif not gt_disc and pred_disc:
            fp += 1
        # else: both false -> true negative, not counted in F1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Edge case: GT has zero discrepancies and so does pred -> perfect.
    if (tp + fn) == 0 and (tp + fp) == 0:
        f1 = 1.0

    return f1, {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -------- credit note handling --------


def score_credit_note(gt: dict, pred: dict) -> float:
    """1.0 only if type, sign convention, and referenced invoice are all correct.

    If ground truth is not a credit note, we only require that the model also
    classified it as an invoice (not credit_note).
    """
    gt_type = _norm_str(gt.get("document_type"))
    pred_type = _norm_str(pred.get("document_type")) if pred else ""
    gt_cn = gt.get("credit_note_handling") or {}
    pred_cn = (pred.get("credit_note_handling") or {}) if pred else {}

    if gt_type != "credit_note":
        # Must not falsely flag as credit_note.
        return 1.0 if pred_type != "credit_note" else 0.0

    # It IS a credit note in GT. All three must pass:
    checks = []
    checks.append(pred_type == "credit_note")
    checks.append(
        _norm_str(gt_cn.get("sign_convention")) == _norm_str(pred_cn.get("sign_convention"))
    )
    gt_ref = _norm_str(gt_cn.get("references_invoice"))
    if gt_ref:
        checks.append(_norm_str(pred_cn.get("references_invoice")) == gt_ref)
    return 1.0 if all(checks) else 0.0


# -------- rebate --------


def score_rebate(
    gt: dict, pred: dict, dkk_tol: float = 1.0, pct_tol: float = 0.01
) -> float:
    g = _norm_num(gt.get("expected_rebate"))
    p = _norm_num(pred.get("expected_rebate")) if pred else None
    if g is None:
        return 1.0  # no rebate in GT, nothing to check
    if p is None:
        return 0.0
    diff = abs(g - p)
    if diff <= dkk_tol:
        return 1.0
    if g != 0 and diff / abs(g) <= pct_tol:
        return 0.5
    return 0.0


# -------- composite --------


def score_invoice(
    gt: dict, pred: dict, *, weights: dict, scoring_cfg: dict
) -> InvoiceScores:
    field_pct, per_field = score_field_extraction(
        gt,
        pred,
        supplier_threshold=scoring_cfg.get("supplier_fuzzy_threshold", 90),
        numeric_tol=scoring_cfg.get("numeric_tolerance", 0.01),
    )
    pm_f1, pm_details = score_price_match(
        gt, pred, desc_threshold=scoring_cfg.get("line_item_description_fuzzy_threshold", 85)
    )
    cn = score_credit_note(gt, pred)
    rb = score_rebate(
        gt,
        pred,
        dkk_tol=scoring_cfg.get("rebate_dkk_tolerance", 1.0),
        pct_tol=scoring_cfg.get("rebate_pct_tolerance", 0.01),
    )
    composite = (
        weights["field_extraction"] * field_pct
        + weights["price_match"] * pm_f1
        + weights["credit_note"] * cn
        + weights["rebate"] * rb
    )
    return InvoiceScores(
        field_extraction_pct=field_pct,
        price_match_pct=pm_f1,
        credit_note_pct=cn,
        rebate_pct=rb,
        composite=composite,
        per_field=per_field,
        price_match_details=pm_details,
    )

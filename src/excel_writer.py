"""Write the 4-sheet evaluation Excel report."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def _write_header(ws, headers: list[str]) -> None:
    for col, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=col, value=h)
        c.fill = HEADER_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(horizontal="left", vertical="center")
    ws.freeze_panes = "A2"


def _autosize(ws, headers: list[str], max_width: int = 60) -> None:
    for col_idx, header in enumerate(headers, start=1):
        col_letter = get_column_letter(col_idx)
        longest = len(str(header))
        for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
            for cell in row:
                v = cell.value
                if v is None:
                    continue
                s = str(v)
                # For JSON columns, clamp.
                if "\n" in s:
                    s = s.split("\n")[0]
                longest = max(longest, len(s))
        ws.column_dimensions[col_letter].width = min(max_width, longest + 2)


def _pct(v: float) -> float:
    return round(v * 100, 2)


def build_output_path(results_dir: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    p = Path(results_dir) / f"eval_{ts}.xlsx"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_report(
    output_path: Path,
    summary_rows: list[dict],
    per_invoice_rows: list[dict],
    raw_rows: list[dict],
) -> None:
    wb = Workbook()

    # Sheet 1: Summary
    ws = wb.active
    ws.title = "Summary"
    headers = [
        "config_name",
        "docs_tested",
        "field_extraction_accuracy_%",
        "price_match_accuracy_%",
        "credit_note_accuracy_%",
        "rebate_accuracy_%",
        "composite_score_%",
        "avg_latency_sec",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost_usd",
        "cost_per_doc_usd",
        "failures",
    ]
    _write_header(ws, headers)
    for r in summary_rows:
        ws.append(
            [
                r["config_name"],
                r["docs_tested"],
                _pct(r["field_extraction_accuracy"]),
                _pct(r["price_match_accuracy"]),
                _pct(r["credit_note_accuracy"]),
                _pct(r["rebate_accuracy"]),
                _pct(r["composite_score"]),
                round(r["avg_latency_sec"], 2),
                r["total_input_tokens"],
                r["total_output_tokens"],
                round(r["total_cost_usd"], 4),
                round(r["cost_per_doc_usd"], 4),
                r["failures"],
            ]
        )
    _autosize(ws, headers)

    # Sheet 2: Per-Invoice
    ws = wb.create_sheet("Per-Invoice")
    headers = [
        "invoice_id",
        "config_name",
        "field_extraction_%",
        "price_match_%",
        "credit_note_%",
        "rebate_%",
        "composite_%",
        "latency_sec",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "notes",
    ]
    _write_header(ws, headers)
    for r in per_invoice_rows:
        ws.append(
            [
                r["invoice_id"],
                r["config_name"],
                _pct(r["field_extraction"]),
                _pct(r["price_match"]),
                _pct(r["credit_note"]),
                _pct(r["rebate"]),
                _pct(r["composite"]),
                round(r["latency_sec"], 2),
                r["input_tokens"],
                r["output_tokens"],
                round(r["cost_usd"], 4),
                r.get("notes", ""),
            ]
        )
    _autosize(ws, headers)

    # Sheet 3: Per-Field (aggregated from per_invoice_rows[...]["per_field"])
    ws = wb.create_sheet("Per-Field")
    headers = [
        "config_name",
        "field",
        "total_attempts",
        "exact_match",
        "partial_match",
        "wrong",
        "accuracy_pct",
    ]
    _write_header(ws, headers)

    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "exact": 0, "partial": 0, "wrong": 0, "sum": 0.0}
    )
    for row in per_invoice_rows:
        cfg = row["config_name"]
        for f in row.get("per_field", []):
            k = (cfg, f["field"])
            agg[k]["attempts"] += 1
            s = f["score"]
            agg[k]["sum"] += s
            if s >= 1.0:
                agg[k]["exact"] += 1
            elif s > 0:
                agg[k]["partial"] += 1
            else:
                agg[k]["wrong"] += 1
    for (cfg, fname), d in sorted(agg.items()):
        acc = d["sum"] / d["attempts"] if d["attempts"] else 0.0
        ws.append([cfg, fname, d["attempts"], d["exact"], d["partial"], d["wrong"], _pct(acc)])
    _autosize(ws, headers)

    # Sheet 4: Raw-Outputs
    ws = wb.create_sheet("Raw-Outputs")
    headers = ["invoice_id", "config_name", "ground_truth_json", "model_output_json", "diff"]
    _write_header(ws, headers)
    for r in raw_rows:
        ws.append(
            [
                r["invoice_id"],
                r["config_name"],
                r["ground_truth_json"],
                r["model_output_json"],
                r["diff"],
            ]
        )
    # JSON columns: wrap text and give them room.
    for col_idx in (3, 4, 5):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 60
        for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col_idx in (1, 2):
        ws.column_dimensions[get_column_letter(col_idx)].width = 22

    wb.save(output_path)


def json_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def json_diff(gt: dict, pred: dict) -> str:
    """Compact per-field diff. Only shows keys where GT != pred."""
    lines = []
    keys = set((gt or {}).keys()) | set((pred or {}).keys())
    for k in sorted(keys):
        g = (gt or {}).get(k)
        p = (pred or {}).get(k)
        if g != p:
            lines.append(f"{k}:\n  gt:   {json.dumps(g, ensure_ascii=False, default=str)}\n  pred: {json.dumps(p, ensure_ascii=False, default=str)}")
    return "\n".join(lines) if lines else "(match)"

"""Main evaluation loop."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openpyxl import load_workbook
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .clients import ModelCall, parse_model_json
from .configs import ModelConfig
from .excel_writer import build_output_path, json_diff, json_pretty, write_report
from .pricing import cost_usd
from .prompts import (
    build_hybrid_extraction_prompt,
    build_hybrid_reasoning_prompt,
    build_unified_prompt,
)
from .schema import ExtractedInvoice
from .scorer import InvoiceScores, score_invoice

log = logging.getLogger(__name__)
console = Console()


@dataclass
class InvoiceJob:
    invoice_id: str
    pdf_path: Path
    agreement_path: Optional[Path]
    agreement_text: str
    ground_truth: dict


def load_jobs(cfg: dict, limit: Optional[int]) -> list[InvoiceJob]:
    invoices_dir = Path(cfg["paths"]["invoices_dir"])
    agreements_dir = Path(cfg["paths"]["agreements_dir"])
    gt_path = Path(cfg["paths"]["ground_truth"])

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_path}. "
            "Ivan/Marius must provide this. Do not synthesize."
        )
    if not invoices_dir.exists():
        raise FileNotFoundError(f"Invoices directory not found: {invoices_dir}")

    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

    jobs: list[InvoiceJob] = []
    for invoice_id, gt in ground_truth.items():
        pdf_path = invoices_dir / invoice_id
        if not pdf_path.exists():
            log.warning("PDF listed in ground truth but missing on disk: %s", pdf_path)
            continue
        agreement_name = gt.get("agreement_file")
        agreement_path = agreements_dir / agreement_name if agreement_name else None
        agreement_text = ""
        if agreement_path and agreement_path.exists():
            agreement_text = _xlsx_to_text(agreement_path)
        elif agreement_name:
            log.warning("Agreement file missing: %s", agreement_path)
        jobs.append(
            InvoiceJob(
                invoice_id=invoice_id,
                pdf_path=pdf_path,
                agreement_path=agreement_path,
                agreement_text=agreement_text,
                ground_truth=gt,
            )
        )
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def _xlsx_to_text(path: Path) -> str:
    """Flatten an .xlsx into plain text for inclusion in the prompt."""
    wb = load_workbook(path, data_only=True, read_only=True)
    out: list[str] = []
    for sheet in wb.worksheets:
        out.append(f"## Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            if all(v is None or v == "" for v in row):
                continue
            out.append("\t".join("" if v is None else str(v) for v in row))
    wb.close()
    return "\n".join(out)


def _validate_model_output(raw_text: str) -> tuple[Optional[dict], Optional[str]]:
    """Parse JSON and validate against schema. Returns (dict, error)."""
    if not raw_text or not raw_text.strip():
        return None, "empty response"
    try:
        obj = parse_model_json(raw_text)
    except Exception as e:
        return None, f"json parse: {e}"
    try:
        validated = ExtractedInvoice.model_validate(obj)
    except ValidationError as e:
        return obj, f"schema: {e.error_count()} errors"
    return validated.model_dump(mode="json"), None


def run_one(
    job: InvoiceJob, cfg: ModelConfig, clients: dict, cfg_yaml: dict
) -> dict:
    """Run a single (invoice, config) pair. Always returns a row dict."""
    pdf_bytes = job.pdf_path.read_bytes()
    agreement_name = job.ground_truth.get("agreement_file") or "N/A"
    total_in = total_out = 0
    total_latency = 0.0
    cost_by_key = 0.0
    notes: list[str] = []
    pred: Optional[dict] = None

    if not cfg.is_hybrid:
        prompt = build_unified_prompt(agreement_name, job.agreement_text)
        client = clients[cfg.extraction.provider]
        call: ModelCall = client.extract_from_pdf(
            cfg_yaml["models"][cfg.extraction.model_key], pdf_bytes, prompt
        )
        total_in += call.input_tokens
        total_out += call.output_tokens
        total_latency += call.latency_sec
        cost_by_key += cost_usd(
            cfg.extraction.model_key,
            call.input_tokens,
            call.output_tokens,
            cfg_yaml["pricing"],
        )
        if call.error:
            notes.append(f"extract error: {call.error}")
        else:
            pred, err = _validate_model_output(call.raw_text)
            if err:
                notes.append(err)
    else:
        # Stage 1: extraction on Gemini (no agreement)
        extract_prompt = build_hybrid_extraction_prompt()
        ext_client = clients[cfg.extraction.provider]
        call1 = ext_client.extract_from_pdf(
            cfg_yaml["models"][cfg.extraction.model_key], pdf_bytes, extract_prompt
        )
        total_in += call1.input_tokens
        total_out += call1.output_tokens
        total_latency += call1.latency_sec
        cost_by_key += cost_usd(
            cfg.extraction.model_key,
            call1.input_tokens,
            call1.output_tokens,
            cfg_yaml["pricing"],
        )
        if call1.error:
            notes.append(f"extract error: {call1.error}")
            pred = None
        else:
            extracted, err = _validate_model_output(call1.raw_text)
            if err:
                notes.append(f"extract {err}")
            if extracted is None:
                pred = None
            else:
                # Stage 2: reasoning on Claude
                reason_prompt = build_hybrid_reasoning_prompt(
                    json.dumps(extracted, ensure_ascii=False, indent=2),
                    agreement_name,
                    job.agreement_text,
                )
                reason_client = clients[cfg.reasoning.provider]
                call2 = reason_client.reason_text(
                    cfg_yaml["models"][cfg.reasoning.model_key], reason_prompt
                )
                total_in += call2.input_tokens
                total_out += call2.output_tokens
                total_latency += call2.latency_sec
                cost_by_key += cost_usd(
                    cfg.reasoning.model_key,
                    call2.input_tokens,
                    call2.output_tokens,
                    cfg_yaml["pricing"],
                )
                if call2.error:
                    notes.append(f"reason error: {call2.error}")
                    pred = extracted  # fall back to extracted-only
                else:
                    pred2, err2 = _validate_model_output(call2.raw_text)
                    if err2:
                        notes.append(f"reason {err2}")
                    pred = pred2 if pred2 is not None else extracted

    scores: InvoiceScores = score_invoice(
        job.ground_truth,
        pred or {},
        weights=cfg_yaml["scoring_weights"],
        scoring_cfg=cfg_yaml["scoring"],
    )

    return {
        "invoice_id": job.invoice_id,
        "config_name": cfg.name,
        "field_extraction": scores.field_extraction_pct,
        "price_match": scores.price_match_pct,
        "credit_note": scores.credit_note_pct,
        "rebate": scores.rebate_pct,
        "composite": scores.composite,
        "latency_sec": total_latency,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": cost_by_key,
        "notes": "; ".join(notes),
        "per_field": [
            {"field": f.field, "score": f.score} for f in scores.per_field
        ],
        "pred": pred,
        "failed": pred is None,
    }


def run(
    cfg_yaml: dict,
    configs: list[ModelConfig],
    clients: dict,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    jobs = load_jobs(cfg_yaml, limit)
    if not jobs:
        raise RuntimeError(
            "No invoice jobs to run. Check that PDFs referenced in ground_truth.json exist."
        )

    console.print(
        f"[bold]Plan:[/bold] {len(jobs)} invoice(s) x {len(configs)} config(s) = "
        f"{len(jobs) * len(configs)} total runs."
    )
    for c in configs:
        console.print(
            f"  - {c.name:<22} extraction={c.extraction.model_key:<20} "
            f"reasoning={c.reasoning.model_key}"
        )
    console.print("  Invoices:")
    for j in jobs:
        console.print(f"    - {j.invoice_id}  (agreement: {j.agreement_path or 'none'})")

    if dry_run:
        console.print("[yellow]--dry-run: no API calls made.[/yellow]")
        return None

    per_invoice_rows: list[dict] = []
    raw_rows: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(jobs) * len(configs))
        for cfg in configs:
            for job in jobs:
                progress.update(task, description=f"{cfg.name} :: {job.invoice_id}")
                row = run_one(job, cfg, clients, cfg_yaml)
                per_invoice_rows.append(row)
                raw_rows.append(
                    {
                        "invoice_id": job.invoice_id,
                        "config_name": cfg.name,
                        "ground_truth_json": json_pretty(job.ground_truth),
                        "model_output_json": json_pretty(row.get("pred") or {}),
                        "diff": json_diff(job.ground_truth, row.get("pred") or {}),
                    }
                )
                progress.advance(task)

    summary_rows = _summarize(per_invoice_rows, configs)
    out_path = build_output_path(cfg_yaml["paths"]["results_dir"])
    write_report(out_path, summary_rows, per_invoice_rows, raw_rows)
    console.print(f"[green]Wrote[/green] {out_path}")
    return out_path


def _summarize(per_invoice_rows: list[dict], configs: list[ModelConfig]) -> list[dict]:
    summary: list[dict] = []
    for cfg in configs:
        rows = [r for r in per_invoice_rows if r["config_name"] == cfg.name]
        if not rows:
            continue
        n = len(rows)
        failures = sum(1 for r in rows if r["failed"])
        scored = [r for r in rows if not r["failed"]] or rows
        ns = len(scored)

        def _avg(field: str) -> float:
            return sum(r[field] for r in scored) / ns if ns else 0.0

        total_cost = sum(r["cost_usd"] for r in rows)
        total_in = sum(r["input_tokens"] for r in rows)
        total_out = sum(r["output_tokens"] for r in rows)
        avg_lat = sum(r["latency_sec"] for r in rows) / n

        summary.append(
            {
                "config_name": cfg.name,
                "docs_tested": n,
                "field_extraction_accuracy": _avg("field_extraction"),
                "price_match_accuracy": _avg("price_match"),
                "credit_note_accuracy": _avg("credit_note"),
                "rebate_accuracy": _avg("rebate"),
                "composite_score": _avg("composite"),
                "avg_latency_sec": avg_lat,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_cost_usd": total_cost,
                "cost_per_doc_usd": total_cost / n if n else 0.0,
                "failures": failures,
            }
        )
    return summary

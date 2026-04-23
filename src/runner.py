"""Main evaluation loop."""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

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
    PromptTemplates,
    build_hybrid_extraction_prompt,
    build_hybrid_reasoning_prompt,
    build_unified_prompt,
)
from .schema import ExtractedInvoice
from .scorer import InvoiceScores, score_invoice

log = logging.getLogger(__name__)
console = Console()


SUPPORTED_INVOICE_EXTS = {".pdf", ".xlsx"}
SUPPORTED_AGREEMENT_EXTS = {".pdf", ".xlsx"}


@dataclass
class InvoiceJob:
    invoice_id: str
    invoice_path: Path
    agreement_paths: list[Path] = field(default_factory=list)
    ground_truth: Optional[dict] = None  # None when no GT provided


def load_jobs(cfg: dict, limit: Optional[int]) -> list[InvoiceJob]:
    """Discover jobs.

    If ground_truth.json exists, it's the source of truth (which invoices, and
    which agreement each one uses). Otherwise: all files in invoices_dir
    become jobs and every job gets every agreement as context (manual
    verification mode).
    """
    invoices_dir = Path(cfg["paths"]["invoices_dir"])
    agreements_dir = Path(cfg["paths"]["agreements_dir"])
    gt_path = Path(cfg["paths"]["ground_truth"])

    log.info(
        "load_jobs: invoices_dir=%s agreements_dir=%s ground_truth=%s",
        invoices_dir.resolve(),
        agreements_dir.resolve(),
        gt_path.resolve(),
    )

    if not invoices_dir.exists():
        raise FileNotFoundError(
            f"Invoices directory not found: {invoices_dir.resolve()}"
        )

    all_agreements = []
    if agreements_dir.exists():
        for ext in SUPPORTED_AGREEMENT_EXTS:
            all_agreements.extend(sorted(agreements_dir.glob(f"*{ext}")))

    jobs: list[InvoiceJob] = []

    if gt_path.exists():
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        for invoice_id, gt in ground_truth.items():
            invoice_path = invoices_dir / invoice_id
            if not invoice_path.exists():
                log.warning(
                    "Invoice listed in ground truth but missing on disk: %s",
                    invoice_path,
                )
                continue
            agreement_name = gt.get("agreement_file")
            agreement_paths: list[Path] = []
            if agreement_name:
                ap = agreements_dir / agreement_name
                if ap.exists():
                    agreement_paths.append(ap)
                else:
                    log.warning("Agreement file missing: %s", ap)
            jobs.append(
                InvoiceJob(
                    invoice_id=invoice_id,
                    invoice_path=invoice_path,
                    agreement_paths=agreement_paths,
                    ground_truth=gt,
                )
            )
    else:
        # No ground truth: every file in invoices_dir becomes a job, using all
        # agreements as context. User verifies manually.
        discovered: list[Path] = []
        for ext in SUPPORTED_INVOICE_EXTS:
            discovered.extend(sorted(invoices_dir.glob(f"*{ext}")))
        if not discovered:
            on_disk = [p.name for p in invoices_dir.iterdir() if p.is_file()][:20]
            raise FileNotFoundError(
                f"No invoice files found in {invoices_dir.resolve()} "
                f"(looking for {sorted(SUPPORTED_INVOICE_EXTS)}). "
                f"Files currently in that directory: {on_disk or '(empty)'}. "
                "Upload some invoices first. "
                "Note: on free-tier hosts with ephemeral disks, uploads are "
                "wiped when the service restarts or re-deploys. If that's "
                "likely what happened, just upload again and start the run "
                "immediately after."
            )
        for inv_path in discovered:
            jobs.append(
                InvoiceJob(
                    invoice_id=inv_path.name,
                    invoice_path=inv_path,
                    agreement_paths=list(all_agreements),
                    ground_truth=None,
                )
            )

    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def _xlsx_to_text(path: Path) -> str:
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


def _load_invoice_content(path: Path) -> tuple[Optional[bytes], Optional[str]]:
    """Return (pdf_bytes, xlsx_text). Exactly one is non-None."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return path.read_bytes(), None
    if ext == ".xlsx":
        return None, _xlsx_to_text(path)
    raise ValueError(f"Unsupported invoice extension: {path}")


def _load_agreement_content(
    paths: list[Path],
) -> tuple[list[bytes], str]:
    """Return (list of PDF bytes, concatenated xlsx text)."""
    pdfs: list[bytes] = []
    text_parts: list[str] = []
    for p in paths:
        ext = p.suffix.lower()
        if ext == ".pdf":
            pdfs.append(p.read_bytes())
        elif ext == ".xlsx":
            text_parts.append(f"### {p.name}\n{_xlsx_to_text(p)}")
    return pdfs, "\n\n".join(text_parts)


def _validate_model_output(raw_text: str) -> tuple[Optional[dict], Optional[str]]:
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


def _tok(n) -> int:
    """Defensive int coercion for token counts from SDK responses."""
    try:
        return int(n) if n is not None else 0
    except (TypeError, ValueError):
        return 0


def _lat(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def run_one(
    job: InvoiceJob,
    cfg: ModelConfig,
    clients: dict,
    cfg_yaml: dict,
    templates: Optional[PromptTemplates] = None,
) -> dict:
    """Run a single (invoice, config) pair. Always returns a row dict."""
    invoice_pdf, invoice_text = _load_invoice_content(job.invoice_path)
    agreement_pdfs, agreement_text = _load_agreement_content(job.agreement_paths)
    has_agreement = bool(agreement_pdfs) or bool(agreement_text)

    total_in = total_out = 0
    total_latency = 0.0
    total_cost = 0.0
    notes: list[str] = []
    pred: Optional[dict] = None

    def _accumulate(call: ModelCall, model_key: str) -> None:
        nonlocal total_in, total_out, total_latency, total_cost
        total_in += _tok(call.input_tokens)
        total_out += _tok(call.output_tokens)
        total_latency += _lat(call.latency_sec)
        total_cost += cost_usd(
            model_key,
            _tok(call.input_tokens),
            _tok(call.output_tokens),
            cfg_yaml["pricing"],
        )

    if not cfg.is_hybrid:
        prompt = build_unified_prompt(
            invoice_text=invoice_text,
            agreement_text=agreement_text or None,
            has_invoice_pdf=invoice_pdf is not None,
            agreement_pdf_count=len(agreement_pdfs),
            with_agreement=has_agreement,
            templates=templates,
        )
        attachments: list[bytes] = []
        if invoice_pdf is not None:
            attachments.append(invoice_pdf)
        attachments.extend(agreement_pdfs)

        client = clients[cfg.extraction.provider]
        call: ModelCall = client.call(
            cfg_yaml["models"][cfg.extraction.model_key], prompt, attachments
        )
        _accumulate(call, cfg.extraction.model_key)
        if call.error:
            notes.append(f"extract error: {call.error}")
        else:
            pred, err = _validate_model_output(call.raw_text)
            if err:
                notes.append(err)
    else:
        # Stage 1: extraction (invoice only, no agreement)
        extract_prompt = build_hybrid_extraction_prompt(
            invoice_text=invoice_text,
            has_invoice_pdf=invoice_pdf is not None,
            templates=templates,
        )
        extract_attachments = [invoice_pdf] if invoice_pdf is not None else []
        ext_client = clients[cfg.extraction.provider]
        call1 = ext_client.call(
            cfg_yaml["models"][cfg.extraction.model_key],
            extract_prompt,
            extract_attachments,
        )
        _accumulate(call1, cfg.extraction.model_key)
        if call1.error:
            notes.append(f"extract error: {call1.error}")
            extracted: Optional[dict] = None
        else:
            extracted, err = _validate_model_output(call1.raw_text)
            if err:
                notes.append(f"extract {err}")

        if extracted is None:
            pred = None
        elif not has_agreement:
            # Nothing to reason about; return the extracted result as-is.
            pred = extracted
        else:
            # Stage 2: reasoning
            reason_prompt = build_hybrid_reasoning_prompt(
                invoice_json=json.dumps(extracted, ensure_ascii=False, indent=2),
                agreement_text=agreement_text or None,
                agreement_pdf_count=len(agreement_pdfs),
                templates=templates,
            )
            reason_client = clients[cfg.reasoning.provider]
            call2 = reason_client.call(
                cfg_yaml["models"][cfg.reasoning.model_key],
                reason_prompt,
                agreement_pdfs,
            )
            _accumulate(call2, cfg.reasoning.model_key)
            if call2.error:
                notes.append(f"reason error: {call2.error}")
                pred = extracted
            else:
                pred2, err2 = _validate_model_output(call2.raw_text)
                if err2:
                    notes.append(f"reason {err2}")
                pred = pred2 if pred2 is not None else extracted

    # Score only if ground truth is available.
    scores: Optional[InvoiceScores] = None
    if job.ground_truth is not None:
        scores = score_invoice(
            job.ground_truth,
            pred or {},
            weights=cfg_yaml["scoring_weights"],
            scoring_cfg=cfg_yaml["scoring"],
        )

    return {
        "invoice_id": job.invoice_id,
        "config_name": cfg.name,
        "field_extraction": scores.field_extraction_pct if scores else None,
        "price_match": scores.price_match_pct if scores else None,
        "credit_note": scores.credit_note_pct if scores else None,
        "rebate": scores.rebate_pct if scores else None,
        "composite": scores.composite if scores else None,
        "latency_sec": total_latency,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": total_cost,
        "notes": "; ".join(notes),
        "per_field": [
            {"field": f.field, "score": f.score}
            for f in (scores.per_field if scores else [])
        ],
        "pred": pred,
        "ground_truth": job.ground_truth,
        "failed": pred is None,
        "has_ground_truth": job.ground_truth is not None,
    }


def run(
    cfg_yaml: dict,
    configs: list[ModelConfig],
    clients: dict,
    limit: Optional[int] = None,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> Optional[Path]:
    jobs = load_jobs(cfg_yaml, limit)
    if not jobs:
        raise RuntimeError(
            "No invoice jobs to run. Upload some invoice PDFs or .xlsx files first."
        )
    total = len(jobs) * len(configs)
    use_web = progress_callback is not None
    has_any_gt = any(j.ground_truth is not None for j in jobs)

    # Snapshot prompt templates at run start so mid-run edits don't mix.
    prompts_path = cfg_yaml.get("paths", {}).get("prompts_file", "data/prompts.json")
    templates = PromptTemplates.load(prompts_path)

    def emit(event: dict) -> None:
        if progress_callback is not None:
            progress_callback(event)

    emit(
        {
            "type": "plan",
            "jobs": [j.invoice_id for j in jobs],
            "configs": [c.name for c in configs],
            "total": total,
            "scoring_enabled": has_any_gt,
        }
    )
    if not use_web:
        mode = "with ground-truth scoring" if has_any_gt else "extraction-only (no GT)"
        console.print(
            f"[bold]Plan:[/bold] {len(jobs)} invoice(s) x {len(configs)} config(s) = "
            f"{total} total runs ({mode})."
        )
        for c in configs:
            console.print(
                f"  - {c.name:<22} extraction={c.extraction.model_key:<20} "
                f"reasoning={c.reasoning.model_key}"
            )
        console.print("  Invoices:")
        for j in jobs:
            ag = ", ".join(p.name for p in j.agreement_paths) or "none"
            console.print(f"    - {j.invoice_id}  (agreements: {ag})")

    if dry_run:
        emit({"type": "log", "message": "Dry-run: plan validated, no API calls made."})
        emit({"type": "done", "output_path": None})
        if not use_web:
            console.print("[yellow]--dry-run: no API calls made.[/yellow]")
        return None

    per_invoice_rows: list[dict] = []
    raw_rows: list[dict] = []
    emit({"type": "start", "total": total})

    def _record(cfg_name: str, job: InvoiceJob, row: dict) -> None:
        per_invoice_rows.append(row)
        raw_rows.append(
            {
                "invoice_id": job.invoice_id,
                "config_name": cfg_name,
                "ground_truth_json": json_pretty(job.ground_truth) if job.ground_truth else "(no ground truth)",
                "model_output_json": json_pretty(row.get("pred") or {}),
                "diff": json_diff(job.ground_truth or {}, row.get("pred") or {}) if job.ground_truth else "(no ground truth — verify manually)",
            }
        )

    def _safe_run_one(job: InvoiceJob, cfg: ModelConfig) -> dict:
        """Never raises. On any exception, returns a failure row so the batch continues."""
        try:
            return run_one(job, cfg, clients, cfg_yaml, templates=templates)
        except Exception as e:
            tb = traceback.format_exc()
            log.exception("run_one failed for %s :: %s", cfg.name, job.invoice_id)
            return {
                "invoice_id": job.invoice_id,
                "config_name": cfg.name,
                "field_extraction": None,
                "price_match": None,
                "credit_note": None,
                "rebate": None,
                "composite": None,
                "latency_sec": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "notes": f"internal error: {type(e).__name__}: {e}",
                "per_field": [],
                "pred": None,
                "ground_truth": job.ground_truth,
                "failed": True,
                "has_ground_truth": job.ground_truth is not None,
                "_traceback": tb,
            }

    if use_web:
        completed = 0
        for cfg in configs:
            for job in jobs:
                emit(
                    {
                        "type": "progress",
                        "completed": completed,
                        "total": total,
                        "current": f"{cfg.name} :: {job.invoice_id}",
                    }
                )
                row = _safe_run_one(job, cfg)
                _record(cfg.name, job, row)
                completed += 1
                if row.get("_traceback"):
                    emit({"type": "log", "message": f"TRACEBACK {cfg.name} :: {job.invoice_id}\n{row['_traceback']}"})
                emit(
                    {
                        "type": "completed",
                        "completed": completed,
                        "total": total,
                        "invoice_id": job.invoice_id,
                        "config_name": cfg.name,
                        "composite": row["composite"],
                        "notes": row["notes"],
                        "has_ground_truth": row["has_ground_truth"],
                    }
                )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating", total=total)
            for cfg in configs:
                for job in jobs:
                    progress.update(task, description=f"{cfg.name} :: {job.invoice_id}")
                    row = _safe_run_one(job, cfg)
                    _record(cfg.name, job, row)
                    progress.advance(task)

    summary_rows = _summarize(per_invoice_rows, configs, has_any_gt)
    out_path = build_output_path(cfg_yaml["paths"]["results_dir"])
    write_report(
        out_path,
        summary_rows,
        per_invoice_rows,
        raw_rows,
        scoring_enabled=has_any_gt,
    )
    emit(
        {
            "type": "done",
            "output_path": str(out_path),
            "summary": summary_rows,
            "scoring_enabled": has_any_gt,
        }
    )
    if not use_web:
        console.print(f"[green]Wrote[/green] {out_path}")
    return out_path


def _summarize(
    per_invoice_rows: list[dict],
    configs: list[ModelConfig],
    has_any_gt: bool,
) -> list[dict]:
    summary: list[dict] = []
    for cfg in configs:
        rows = [r for r in per_invoice_rows if r["config_name"] == cfg.name]
        if not rows:
            continue
        n = len(rows)
        failures = sum(1 for r in rows if r["failed"])
        scored = [r for r in rows if not r["failed"] and r["has_ground_truth"]]
        ns = len(scored)

        def _avg(field: str) -> Optional[float]:
            if not ns:
                return None
            return sum(r[field] for r in scored) / ns

        total_cost = sum(r["cost_usd"] for r in rows)
        total_in = sum(r["input_tokens"] for r in rows)
        total_out = sum(r["output_tokens"] for r in rows)
        avg_lat = sum(r["latency_sec"] for r in rows) / n if n else 0.0

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

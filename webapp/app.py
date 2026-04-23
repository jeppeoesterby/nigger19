"""Flask web UI for the evaluation harness.

Wraps the existing runner so Ivan can trigger runs from a browser, watch
live progress, and download the Excel report.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    get_flashed_messages,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from src.clients import build_clients
from src.configs import CONFIGS, filter_configs
from src.runner import load_jobs, run

log = logging.getLogger(__name__)

# In-memory run registry. Fine for a local single-user tool.
RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()
MAX_LOG_LINES = 500


def create_app(config_path: str = "config.yaml") -> Flask:
    load_dotenv()
    app = Flask(__name__)
    app.config["CFG_PATH"] = config_path
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB per request
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(16)

    def _cfg() -> dict:
        return yaml.safe_load(Path(app.config["CFG_PATH"]).read_text(encoding="utf-8"))

    def _ensure_data_dirs() -> None:
        c = _cfg()
        for key in ("invoices_dir", "agreements_dir", "results_dir"):
            Path(c["paths"][key]).mkdir(parents=True, exist_ok=True)

    _ensure_data_dirs()

    def _scan_data(cfg_yaml: dict) -> dict:
        invoices_dir = Path(cfg_yaml["paths"]["invoices_dir"])
        agreements_dir = Path(cfg_yaml["paths"]["agreements_dir"])
        gt_path = Path(cfg_yaml["paths"]["ground_truth"])

        info = {
            "invoices_dir": str(invoices_dir),
            "agreements_dir": str(agreements_dir),
            "ground_truth_path": str(gt_path),
            "ground_truth_exists": gt_path.exists(),
            "invoice_count_on_disk": 0,
            "agreement_count_on_disk": 0,
            "jobs": [],
            "error": None,
            "keys_configured": {
                "anthropic": bool(__import__("os").environ.get("ANTHROPIC_API_KEY")),
                "google": bool(__import__("os").environ.get("GOOGLE_API_KEY")),
            },
        }
        if invoices_dir.exists():
            info["invoice_count_on_disk"] = sum(
                1
                for p in invoices_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".pdf", ".xlsx"}
            )
        if agreements_dir.exists():
            info["agreement_count_on_disk"] = sum(
                1
                for p in agreements_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".pdf", ".xlsx"}
            )
        if gt_path.exists():
            try:
                jobs = load_jobs(cfg_yaml, limit=None)
                info["jobs"] = [
                    {
                        "id": j.invoice_id,
                        "agreement": j.agreement_path.name if j.agreement_path else None,
                    }
                    for j in jobs
                ]
            except Exception as e:
                info["error"] = str(e)
        return info

    def _list_past_results(cfg_yaml: dict) -> list[dict]:
        results_dir = Path(cfg_yaml["paths"]["results_dir"])
        if not results_dir.exists():
            return []
        out = []
        for p in sorted(results_dir.glob("eval_*.xlsx"), reverse=True):
            st = p.stat()
            out.append(
                {
                    "name": p.name,
                    "size_kb": max(1, st.st_size // 1024),
                    "mtime": datetime.fromtimestamp(st.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
            )
        return out[:25]

    # ---------------- routes ----------------

    @app.route("/")
    def index():
        cfg_yaml = _cfg()
        return render_template(
            "index.html",
            configs=[c.name for c in CONFIGS],
            data_info=_scan_data(cfg_yaml),
            past_results=_list_past_results(cfg_yaml),
            cfg=cfg_yaml,
            active_runs=[
                r for r in RUNS.values() if r["status"] in ("running", "starting")
            ],
        )

    @app.route("/start", methods=["POST"])
    def start_run():
        selected_raw = request.form.getlist("configs")
        selected = selected_raw if selected_raw else None
        limit_str = request.form.get("limit", "").strip()
        limit = int(limit_str) if limit_str else None
        dry_run = bool(request.form.get("dry_run"))

        try:
            configs = filter_configs(selected)
        except ValueError as e:
            return f"<p>Invalid config selection: {e}</p>", 400

        run_id = uuid.uuid4().hex[:8]
        state = {
            "id": run_id,
            "status": "starting",
            "log": deque(maxlen=MAX_LOG_LINES),
            "progress": {"completed": 0, "total": 0, "current": ""},
            "output_path": None,
            "output_filename": None,
            "summary": [],
            "error": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config_names": [c.name for c in configs],
            "limit": limit,
            "dry_run": dry_run,
        }
        with RUNS_LOCK:
            RUNS[run_id] = state

        t = threading.Thread(
            target=_execute_run,
            args=(run_id, configs, limit, dry_run, _cfg()),
            daemon=True,
        )
        t.start()
        return redirect(url_for("run_page", run_id=run_id))

    @app.route("/runs/<run_id>")
    def run_page(run_id: str):
        with RUNS_LOCK:
            state = RUNS.get(run_id)
        if not state:
            abort(404)
        return render_template("run.html", run=state)

    @app.route("/runs/<run_id>/status")
    def run_status(run_id: str):
        with RUNS_LOCK:
            state = RUNS.get(run_id)
        if not state:
            return jsonify({"error": "not found"}), 404
        # deque is not JSON-serializable directly; cast to list.
        payload = {
            "id": state["id"],
            "status": state["status"],
            "progress": state["progress"],
            "log": list(state["log"]),
            "output_filename": state["output_filename"],
            "summary": state["summary"],
            "error": state["error"],
            "started_at": state["started_at"],
            "config_names": state["config_names"],
            "limit": state["limit"],
            "dry_run": state["dry_run"],
        }
        return jsonify(payload)

    @app.route("/results")
    def list_results():
        cfg_yaml = _cfg()
        return render_template(
            "results.html", past_results=_list_past_results(cfg_yaml)
        )

    @app.route("/results/<path:filename>")
    def download_result(filename: str):
        cfg_yaml = _cfg()
        results_dir = Path(cfg_yaml["paths"]["results_dir"]).resolve()
        target = (results_dir / filename).resolve()
        if not str(target).startswith(str(results_dir)):
            abort(403)
        if not target.exists():
            abort(404)
        return send_from_directory(results_dir, filename, as_attachment=True)

    # ---------------- data upload ----------------

    def _save_files(files, target_dir: Path, allowed_exts: set[str]) -> tuple[int, int]:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved = skipped = 0
        for f in files:
            if not f or not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in allowed_exts:
                skipped += 1
                continue
            safe = secure_filename(f.filename) or f"upload{ext}"
            f.save(str(target_dir / safe))
            saved += 1
        return saved, skipped

    @app.route("/upload/invoices", methods=["POST"])
    def upload_invoices():
        cfg_yaml = _cfg()
        files = request.files.getlist("files")
        saved, skipped = _save_files(
            files, Path(cfg_yaml["paths"]["invoices_dir"]), {".pdf", ".xlsx"}
        )
        msg = f"Uploaded {saved} invoice file(s)."
        if skipped:
            msg += f" Skipped {skipped} file(s) with unsupported extension (only .pdf, .xlsx)."
        flash(msg)
        return redirect(url_for("index"))

    @app.route("/upload/agreements", methods=["POST"])
    def upload_agreements():
        cfg_yaml = _cfg()
        files = request.files.getlist("files")
        saved, skipped = _save_files(
            files, Path(cfg_yaml["paths"]["agreements_dir"]), {".pdf", ".xlsx"}
        )
        msg = f"Uploaded {saved} agreement file(s)."
        if skipped:
            msg += f" Skipped {skipped} file(s) with unsupported extension (only .pdf, .xlsx)."
        flash(msg)
        return redirect(url_for("index"))

    @app.route("/upload/ground_truth", methods=["POST"])
    def upload_ground_truth():
        cfg_yaml = _cfg()
        f = request.files.get("file")
        gt_path = Path(cfg_yaml["paths"]["ground_truth"])
        if not f or not f.filename:
            flash("No file selected.")
            return redirect(url_for("index"))
        if Path(f.filename).suffix.lower() != ".json":
            flash("Ground truth must be a .json file.")
            return redirect(url_for("index"))
        content = f.read()
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("ground_truth.json must be a JSON object")
        except Exception as e:
            flash(f"Invalid JSON: {e}")
            return redirect(url_for("index"))
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_bytes(content)
        flash(f"Ground truth saved ({len(parsed)} invoice entries).")
        return redirect(url_for("index"))

    @app.route("/clear", methods=["POST"])
    def clear_data():
        cfg_yaml = _cfg()
        what = request.form.get("what", "")
        removed = 0
        if what == "invoices":
            d = Path(cfg_yaml["paths"]["invoices_dir"])
            if d.exists():
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in {".pdf", ".xlsx"}:
                        p.unlink()
                        removed += 1
            flash(f"Deleted {removed} invoice file(s).")
        elif what == "agreements":
            d = Path(cfg_yaml["paths"]["agreements_dir"])
            if d.exists():
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in {".pdf", ".xlsx"}:
                        p.unlink()
                        removed += 1
            flash(f"Deleted {removed} agreement file(s).")
        elif what == "ground_truth":
            p = Path(cfg_yaml["paths"]["ground_truth"])
            if p.exists():
                p.unlink()
                flash("Ground truth deleted.")
            else:
                flash("No ground truth to delete.")
        elif what == "all":
            for key in ("invoices_dir", "agreements_dir"):
                d = Path(cfg_yaml["paths"][key])
                if d.exists():
                    shutil.rmtree(d)
                    d.mkdir(parents=True, exist_ok=True)
            p = Path(cfg_yaml["paths"]["ground_truth"])
            if p.exists():
                p.unlink()
            flash("All uploaded data cleared.")
        else:
            flash("Unknown clear target.")
        return redirect(url_for("index"))

    return app


def _execute_run(
    run_id: str,
    configs: list,
    limit: Optional[int],
    dry_run: bool,
    cfg_yaml: dict,
) -> None:
    def emit(event: dict) -> None:
        with RUNS_LOCK:
            state = RUNS.get(run_id)
            if not state:
                return
            t = event.get("type")
            if t == "plan":
                state["log"].append(
                    f"Plan: {len(event['jobs'])} invoices x "
                    f"{len(event['configs'])} configs = {event['total']} runs."
                )
                state["progress"]["total"] = event["total"]
            elif t == "start":
                state["status"] = "running"
                state["log"].append("Starting API calls...")
            elif t == "progress":
                state["progress"]["completed"] = event["completed"]
                state["progress"]["total"] = event["total"]
                state["progress"]["current"] = event["current"]
            elif t == "completed":
                state["log"].append(
                    f"[{event['completed']}/{event['total']}] "
                    f"{event['config_name']} :: {event['invoice_id']}  "
                    f"composite={event['composite']*100:.1f}%"
                    + (f"  ({event['notes']})" if event["notes"] else "")
                )
            elif t == "log":
                state["log"].append(event["message"])
            elif t == "done":
                state["status"] = "done"
                state["output_path"] = event.get("output_path")
                if event.get("output_path"):
                    state["output_filename"] = Path(event["output_path"]).name
                state["summary"] = event.get("summary") or []
                state["log"].append("Done.")
            elif t == "error":
                state["status"] = "error"
                state["error"] = event.get("message")
                state["log"].append(f"ERROR: {event.get('message')}")

    try:
        clients = {} if dry_run else build_clients(cfg_yaml)
        run(
            cfg_yaml=cfg_yaml,
            configs=configs,
            clients=clients,
            limit=limit,
            dry_run=dry_run,
            progress_callback=emit,
        )
    except Exception as e:
        log.exception("Run %s failed", run_id)
        emit({"type": "error", "message": str(e)})

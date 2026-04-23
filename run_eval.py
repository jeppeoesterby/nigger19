#!/usr/bin/env python3
"""DiFacto LLM evaluation harness entry point.

Usage:
    python run_eval.py                                              # all configs, all docs
    python run_eval.py --configs "Claude Sonnet 4.6,Hybrid Sonnet"  # subset
    python run_eval.py --limit 5                                    # first 5 invoices
    python run_eval.py --dry-run                                    # validate + print plan
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console

from src.clients import build_clients
from src.configs import filter_configs
from src.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="DiFacto LLM evaluation harness")
    parser.add_argument(
        "--config-file", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--configs",
        default=None,
        help='Comma-separated config names, e.g. "Claude Sonnet 4.6,Hybrid Sonnet"',
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N invoices"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without calling APIs",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv()

    console = Console()

    cfg_path = Path(args.config_file).resolve()
    if not cfg_path.exists():
        console.print(f"[red]Config file not found:[/red] {cfg_path}")
        return 2
    cfg_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Absolute-ize data paths relative to config file's directory, so the tool
    # works regardless of CWD.
    base_dir = cfg_path.parent
    if isinstance(cfg_yaml.get("paths"), dict):
        for k, v in list(cfg_yaml["paths"].items()):
            if isinstance(v, str) and not Path(v).is_absolute():
                cfg_yaml["paths"][k] = str((base_dir / v).resolve())

    weights = cfg_yaml["scoring_weights"]
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 0.001:
        console.print(
            f"[red]scoring_weights must sum to 1.0 (got {total_w}).[/red] Fix config.yaml."
        )
        return 2

    try:
        configs = filter_configs(args.configs.split(",") if args.configs else None)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    if args.dry_run:
        # Build no clients for dry-run (avoid needing API keys).
        clients = {}
    else:
        try:
            clients = build_clients(cfg_yaml)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return 2

    try:
        run(
            cfg_yaml=cfg_yaml,
            configs=configs,
            clients=clients,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 2
    except Exception as e:
        console.print(f"[red]Fatal:[/red] {e}")
        logging.exception("Fatal error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

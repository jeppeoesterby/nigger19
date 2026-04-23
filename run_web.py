#!/usr/bin/env python3
"""Entry point for the DiFacto LLM evaluation web UI.

Usage:
    python run_web.py                  # 127.0.0.1:5000
    python run_web.py --port 8000
    python run_web.py --host 0.0.0.0   # expose on LAN (careful: your API keys)
"""
from __future__ import annotations

import argparse
import logging
import sys

from webapp.app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="DiFacto LLM Eval web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    parser.add_argument("--config-file", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_app(config_path=args.config_file)
    print(f"DiFacto LLM Eval web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

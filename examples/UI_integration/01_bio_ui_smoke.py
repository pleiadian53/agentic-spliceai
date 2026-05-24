#!/usr/bin/env python
"""Demonstrate + smoke-test the existing Bioinformatics Lab UI (server/bio/).

This is the UI-integration baseline use case: drive the genome-view prediction
API the way the demo does, validate the response, and pre-warm the prediction
cache so the first live click in the browser is instant.

**Current scope — base models only.** The genome view validates ``model``
against ``list_available_models()`` (base models from ``settings.yaml``:
``spliceai``, ``openspliceai``). Meta-model selection (M1-S/M2-S) is the next
phase; this script establishes the working baseline it will extend.

What it does
------------
1. (optional) Launches a temporary ``server.bio.app`` and waits until healthy.
2. ``GET /api/models`` — lists available models; checks the requested one exists.
3. For each demo gene, ``GET /api/genome/{gene}/predict?model=&threshold=`` —
   times the cold call (incl. one-time model load) and a warm (cached) re-call,
   and validates the genome-view response shape (tracks + TP/FP/FN markers).
4. Prints a summary table. Non-zero exit if any gene fails validation.

Usage
-----
    PY=~/miniforge3/envs/agentic-spliceai/bin/python

    # Self-contained: start a temporary server, run, tear down
    $PY examples/UI_integration/01_bio_ui_smoke.py --start-server

    # Against an already-running server (keep it up for a live demo)
    $PY -m server.bio.app                       # -> http://localhost:8005
    $PY examples/UI_integration/01_bio_ui_smoke.py

    # Pre-warm specific genes only
    $PY examples/UI_integration/01_bio_ui_smoke.py --genes BRCA1 SERPINA1
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Demo genes: BRCA1 (familiar cancer gene), TP53 (tumor suppressor),
# SERPINA1 (the COPD use case in the Lab notebooks).
DEFAULT_GENES = ["BRCA1", "TP53", "SERPINA1"]
DEFAULT_BASE_URL = "http://localhost:8005"

# Fields the genome-view response (schemas.py::GenomeResponse) must carry.
REQUIRED_FIELDS = (
    "gene_name", "gene_id", "chrom", "strand", "model", "threshold",
    "positions", "donor_prob", "acceptor_prob",
    "gt_positions", "markers", "n_tp", "n_fp", "n_fn", "total_positions",
)


def http_get_json(url: str, timeout: float) -> tuple[int, object]:
    """GET a URL, return (status_code, parsed_json_or_error_text)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, str(e)
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return 0, str(e)


def wait_for_server(base_url: str, timeout: float) -> bool:
    """Poll /api/models until the server answers or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, _ = http_get_json(f"{base_url}/api/models", timeout=5)
        if status == 200:
            return True
        time.sleep(1.0)
    return False


def start_server() -> subprocess.Popen:
    """Launch ``python -m server.bio.app`` in its own process group.

    Runs from the project root so the ``server.bio`` package resolves, in a new
    session so the uvicorn reloader's child dies with the group on teardown.
    """
    project_root = Path(__file__).resolve().parents[2]
    return subprocess.Popen(
        [sys.executable, "-m", "server.bio.app"],
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate the server process group started by :func:`start_server`."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def validate_genome_response(data: object) -> Optional[str]:
    """Return an error string if the response is not a valid genome view, else None."""
    if not isinstance(data, dict):
        return f"expected JSON object, got {type(data).__name__}"
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return f"missing fields: {missing}"
    n = len(data["positions"])
    if not (len(data["donor_prob"]) == len(data["acceptor_prob"]) == n):
        return "positions / donor_prob / acceptor_prob length mismatch"
    if n == 0:
        return "no positions returned"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate + smoke-test the Bio Lab genome-view UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    parser.add_argument("--model", default="openspliceai",
                        help="Base model to predict with (default: openspliceai)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--start-server", action="store_true",
                        help="Launch a temporary server.bio.app and tear it down after")
    parser.add_argument("--startup-timeout", type=float, default=90.0,
                        help="Seconds to wait for --start-server to become healthy")
    parser.add_argument("--request-timeout", type=float, default=180.0,
                        help="Per-request timeout (cold call includes ~8s model load)")
    args = parser.parse_args()

    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.start_server:
            print(f"Starting server.bio.app (waiting up to {args.startup_timeout:.0f}s)...")
            server_proc = start_server()
            if not wait_for_server(args.base_url, args.startup_timeout):
                print("ERROR: server did not become healthy in time.")
                return 1
            print(f"  Server healthy at {args.base_url}")

        # 1. List models — also our reachability check
        status, models = http_get_json(f"{args.base_url}/api/models", timeout=10)
        if status != 200:
            print(f"ERROR: cannot reach {args.base_url}/api/models ({models}).")
            print("  Start the server first (or pass --start-server):")
            print(f"    {sys.executable} -m server.bio.app   # -> {args.base_url}")
            return 1
        model_names = [m["name"] for m in models]
        print(f"Available models (base only, current UI): {model_names}")
        if args.model not in model_names:
            print(f"ERROR: '{args.model}' is not an available model. "
                  f"Choose one of {model_names}.")
            print("  (Meta models M1-S/M2-S are not wired into the UI yet — see README.)")
            return 1

        # 2. Predict per gene: cold (incl. model load) + warm (cached) timing
        print(f"\n{'gene':<12}{'status':<8}{'positions':>11}{'TP':>5}{'FP':>5}{'FN':>5}"
              f"{'cold(s)':>9}{'warm(s)':>9}")
        print("-" * 64)
        url_tmpl = (f"{args.base_url}/api/genome/{{gene}}/predict"
                    f"?model={args.model}&threshold={args.threshold}")
        n_fail = 0
        for gene in args.genes:
            url = url_tmpl.format(gene=gene)
            t0 = time.time()
            status, data = http_get_json(url, timeout=args.request_timeout)
            cold = time.time() - t0
            if status != 200:
                detail = data.get("detail") if isinstance(data, dict) else data
                print(f"{gene:<12}{'FAIL':<8}  HTTP {status}: {detail}")
                n_fail += 1
                continue
            err = validate_genome_response(data)
            if err:
                print(f"{gene:<12}{'FAIL':<8}  {err}")
                n_fail += 1
                continue
            t0 = time.time()
            http_get_json(url, timeout=args.request_timeout)  # warm (cache hit)
            warm = time.time() - t0
            print(f"{gene:<12}{'OK':<8}{data['total_positions']:>11,}"
                  f"{data['n_tp']:>5}{data['n_fp']:>5}{data['n_fn']:>5}"
                  f"{cold:>9.1f}{warm:>9.2f}")

        print("-" * 64)
        ok = len(args.genes) - n_fail
        print(f"{ok}/{len(args.genes)} genes OK; prediction cache now warm for the demo.")
        return 1 if n_fail else 0

    finally:
        if server_proc is not None:
            print("Stopping temporary server...")
            stop_server(server_proc)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Phase E: warm the Bio Lab UI caches for a live meta-overlay demo.

Hits a RUNNING Bio Lab server's genome endpoint for each demo gene — base-only
and with each meta overlay — so the server's in-memory model + prediction caches
are hot and the first click in the live demo is instant. Doubles as an
end-to-end smoke test of the meta overlay (Phases B–D): it prints base-vs-meta
TP/FP/FN per gene, so you can eyeball that the meta layer recovers base misses
before showing anyone.

Prerequisites
-------------
1. Phase-A feature cache exists for these genes (instant if already warmed):
     python examples/UI_integration/02_build_showcase_feature_cache.py
2. The server is running:
     ~/miniforge3/envs/agentic-spliceai/bin/python -m server.bio.app   # port 8005

Usage
-----
    PY=~/miniforge3/envs/agentic-spliceai/bin/python
    $PY examples/UI_integration/07_warm_ui_cache.py
    $PY examples/UI_integration/07_warm_ui_cache.py --genes UNC13A STMN2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# BRCA1 + ALS panel — the showcase set warmed by 02_build_showcase_feature_cache.py.
DEFAULT_GENES = ["BRCA1", "STMN2", "UNC13A", "SOD1", "TARDBP", "FUS", "C9orf72"]
DEFAULT_META = ["m1s_v3_neuronal", "m2s_v3_neuronal"]


def _get(url: str, timeout: float = 900.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    p = argparse.ArgumentParser(description="Warm + smoke-test the Bio Lab meta overlay")
    p.add_argument("--server", default="http://localhost:8005")
    p.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    p.add_argument("--base-model", default="openspliceai")
    p.add_argument("--meta-models", nargs="+", default=DEFAULT_META)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    base = args.server.rstrip("/")

    print(f"Warming {base} for {len(args.genes)} genes × ({args.base_model} + "
          f"{', '.join(args.meta_models)}) @ threshold {args.threshold}\n")
    ok = fail = 0
    for gene in args.genes:
        g = urllib.parse.quote(gene)
        try:
            t = time.time()
            d = _get(f"{base}/api/genome/{g}/predict?model={args.base_model}&threshold={args.threshold}")
            print(f"[base]  {gene:10s} TP={d['n_tp']:>3} FP={d['n_fp']:>3} FN={d['n_fn']:>3}   ({time.time()-t:5.1f}s)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[base]  {gene:10s} FAILED: {e}")
            fail += 1
        for mm in args.meta_models:
            try:
                t = time.time()
                d = _get(f"{base}/api/genome/{g}/predict?model={args.base_model}&meta={urllib.parse.quote(mm)}&threshold={args.threshold}")
                print(f"  ↳ {mm:18s} {gene:10s} "
                      f"FN {d['n_fn']:>3}→{d['meta_n_fn']:<3} | "
                      f"FP {d['n_fp']:>3}→{d['meta_n_fp']:<4} | "
                      f"meta TP={d['meta_n_tp']:>3}   ({time.time()-t:5.1f}s)")
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ↳ {mm:18s} {gene:10s} FAILED: {e}")
                fail += 1
    print(f"\nWarmed {ok} responses, {fail} failed.")
    if fail:
        print("Hint: ensure the server is running and the Phase-A feature cache "
              "exists (02_build_showcase_feature_cache.py).")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

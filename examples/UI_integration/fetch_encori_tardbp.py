#!/usr/bin/env python
"""Fetch neuronal TDP-43 (TARDBP) CLIP peaks from ENCORI/starBase (hg38) into the
project's eCLIP-peaks schema, to enrich the RBP modality beyond ENCODE K562/HepG2.

Why: ENCODE eCLIP (our `eclip_peaks.parquet`) is K562+HepG2 only, so TARDBP has
zero peaks at neuronal genes like STMN2/UNC13A. ENCORI's RBPTarget API serves
TARDBP clusters from SH-SY5Y (neuronal), H9, HEK293/HEK293T on assembly=hg38, and
they land directly on the ALS cryptic exons (verified 2026-05-21).

Scope: ENCORI's API is per-gene (the genome-wide `target=all` dump times out), so
this is for the showcase/demo + M4 prototype. A genome-wide retrain feed should
use POSTAR3 bulk hg38 BEDs or reprocessed Brown(E-MTAB-11243)/Tollervey(E-MTAB-530)
iCLIP. See reference_tdp43_neuronal_clip_sources in project memory.

Output matches `aggregate_eclip_peaks.py`'s schema:
    chrom, start, end, rbp, cell_line, signal_value, neg_log10_pvalue, strand
- start/end from ENCORI narrowStart/narrowEnd (point crosslink sites widened to ≥1 bp).
- one row per (cluster × cell_line) so the data stays cell-type-resolved.
- signal_value = clipExpNum (supporting CLIP experiments; ENCORI omits p-values here).

Usage:
    PY=~/miniforge3/envs/agentic-spliceai/bin/python
    # Showcase + cryptic genes; neuronal cell lines only:
    $PY examples/UI_integration/fetch_encori_tardbp.py --neuronal-only
    # Merge with the canonical ENCODE parquet into an augmented copy (no overwrite):
    $PY examples/UI_integration/fetch_encori_tardbp.py --neuronal-only --merge
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment

setup_example_environment()

# Genes that matter for the ALS cryptic-splicing demo + the showcase panel.
DEFAULT_GENES = ["STMN2", "UNC13A", "SOD1", "TARDBP", "FUS", "C9orf72", "BRCA1", "TP53"]
NEURONAL_CELL_LINES = {"SH-SY5Y", "H9"}  # neuronal / neural-progenitor-ish, vs HEK293/HEK293T
ENCORI_API = "https://rnasysu.com/encori/api/RBPTarget/"
ECLIP_SCHEMA = ["chrom", "start", "end", "rbp", "cell_line",
                "signal_value", "neg_log10_pvalue", "strand"]


def fetch_encori(target: str = "all", cell_type: str = "all",
                 clip_exp_min: int = 1, timeout: float = 90.0) -> List[dict]:
    """Fetch TARDBP clusters from ENCORI → eCLIP-schema rows.

    target="all" + cell_type=<neuronal> gives a genome-wide neuronal set; a gene
    symbol restricts to that target (cell_type="all" then keeps every cell line).
    """
    params = {
        "assembly": "hg38", "geneType": "mRNA", "RBP": "TARDBP",
        "clipExpNum": str(clip_exp_min), "pancancerNum": "0",
        "target": target, "cellType": cell_type,
    }
    url = ENCORI_API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode()

    rows: List[dict] = []
    header = None
    for line in io.StringIO(text):
        line = line.rstrip("\n")
        if line.startswith("#") or not line:
            continue
        cols = line.split("\t")
        if header is None:
            header = cols
            continue
        rec = dict(zip(header, cols))
        chrom = rec["chromosome"]
        start = int(rec["narrowStart"])
        end = int(rec["narrowEnd"])
        if end <= start:                       # point crosslink site → widen to 1 bp
            end = start + 1
        n_exp = float(rec.get("clipExpNum", 0) or 0)
        # ENCORI aggregates cell lines per cluster ("H9,HEK293,SH-SY5Y") → one row each
        for cl in rec.get("cellline/tissue", "").split(","):
            cl = cl.strip()
            if not cl:
                continue
            rows.append({
                "chrom": chrom, "start": start, "end": end,
                "rbp": "TARDBP", "cell_line": cl,
                "signal_value": n_exp, "neg_log10_pvalue": 0.0, "strand": rec["strand"],
            })
    return rows


def main() -> int:
    import polars as pl

    parser = argparse.ArgumentParser(description="Fetch ENCORI TARDBP peaks (hg38) → eCLIP parquet")
    parser.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    parser.add_argument("--genome-wide", action="store_true",
                        help="Fetch ALL target genes for the neuronal cell lines "
                             f"{sorted(NEURONAL_CELL_LINES)} (target=all per cell type), "
                             "rather than the --genes list. The genome-wide neuronal set.")
    parser.add_argument("--clip-exp-min", type=int, default=1,
                        help="Min supporting CLIP experiments per cluster (ENCORI clipExpNum)")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-request timeout (genome-wide target=all is slow)")
    parser.add_argument("--neuronal-only", action="store_true",
                        help=f"Keep only neuronal cell lines {sorted(NEURONAL_CELL_LINES)}")
    parser.add_argument("--out", type=Path,
                        default=Path("data/mane/GRCh38/rbp_data/encori_tardbp_neuronal.parquet"))
    parser.add_argument("--merge", action="store_true",
                        help="Also write eclip_peaks_augmented.parquet = canonical ENCODE ∪ TARDBP "
                             "(does NOT overwrite the canonical eclip_peaks.parquet)")
    args = parser.parse_args()

    all_rows: List[dict] = []
    if args.genome_wide:
        for ct in sorted(NEURONAL_CELL_LINES):
            try:
                rows = fetch_encori(target="all", cell_type=ct,
                                    clip_exp_min=args.clip_exp_min, timeout=args.timeout)
            except Exception as e:
                print(f"  [warn] genome-wide {ct}: fetch failed ({e})")
                continue
            # multi-cell-line clusters expand to all listed lines; keep neuronal only
            rows = [r for r in rows if r["cell_line"] in NEURONAL_CELL_LINES]
            print(f"  genome-wide {ct:8s}: {len(rows):5d} neuronal peak-rows")
            all_rows.extend(rows)
    else:
        for g in args.genes:
            try:
                rows = fetch_encori(target=g, cell_type="all",
                                    clip_exp_min=args.clip_exp_min, timeout=args.timeout)
            except Exception as e:
                print(f"  [warn] {g}: fetch failed ({e})")
                continue
            if args.neuronal_only:
                rows = [r for r in rows if r["cell_line"] in NEURONAL_CELL_LINES]
            cls = sorted({r["cell_line"] for r in rows})
            print(f"  {g:8s}: {len(rows):3d} peak-rows  cell_lines={cls}")
            all_rows.extend(rows)

    if not all_rows:
        print("No peaks fetched.")
        return 1

    schema = {
        "chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64, "rbp": pl.Utf8,
        "cell_line": pl.Utf8, "signal_value": pl.Float64,
        "neg_log10_pvalue": pl.Float64, "strand": pl.Utf8,
    }
    df = pl.DataFrame(all_rows, schema=schema)
    df = df.unique().sort(["chrom", "start", "end", "cell_line"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"\nWrote {args.out}  ({df.height} rows, {df['rbp'].n_unique()} RBP, "
          f"cell_lines={sorted(df['cell_line'].unique().to_list())})")

    if args.merge:
        canonical = Path("data/mane/GRCh38/rbp_data/eclip_peaks.parquet")
        if not canonical.exists():
            print(f"  [warn] canonical {canonical} missing; skipping merge")
        else:
            base = pl.read_parquet(canonical).select(ECLIP_SCHEMA)
            aug = pl.concat([base, df.select(ECLIP_SCHEMA)]).unique()
            out_aug = canonical.with_name("eclip_peaks_augmented.parquet")
            aug.write_parquet(out_aug)
            print(f"Wrote {out_aug}  (ENCODE {base.height} + TARDBP {df.height} "
                  f"= {aug.height} rows). Point the retrain's eCLIP here; canonical untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Position-level evaluation of variant delta scores on MutSpliceDB.

Every other variant metric in this project is a *magnitude* metric: how big is
the largest delta, and does a rule-based label derived from it match the
observed effect type. Both are aggregations that discard **where** the model put
its signal. This script asks the position-level question instead:

    When a variant disrupts a splice site, does the model's largest loss land
    on **that site**, and is it the **right kind** of site?

That is the same question M3 is scored on (can the model rank the right
*position*, not just the right locus), asked in the variant setting. It is where
both the M3-R negative and the locus-cancellation result predict difficulty.

What the truth is, and what it is not
-------------------------------------
MutSpliceDB does **not** carry an independently measured induced-site
coordinate. Its TSV has one ``position`` per row, which after HGVS resolution is
the **variant** position, plus a categorical ``site_type``
(``intron_retention_region``) and ``effect_type``. A metric of the form
"did the argmax land on the measured induced site" is therefore not available
from this corpus, and any claim to the contrary is over-reading the schema.

What *is* independently derivable, per row, without consulting any model:

``expected_type``
    From the HGVS intronic offset. ``c.N+k`` sits in the intron downstream of an
    exon, so the site it disrupts is a **donor**; ``c.N-k`` sits upstream of an
    exon, so the site is an **acceptor**. Rows with no intronic offset (purely
    exonic variants) name no site and are excluded from the type metric.

``expected_site``
    The nearest annotated MANE splice site of ``expected_type`` in the same
    gene. Read from the MANE splice-site track, which is strand-resolved and
    carries an explicit ``splice_type`` per position.

Both come from the annotation and the variant nomenclature, never from a
prediction, so the comparison is not circular.

.. note::
   Truth comes from ``data/mane/GRCh38/splice_sites_track.parquet`` rather than
   ``SpliceEventDetector.get_gene_structure``. The latter sorts exons by genomic
   start and drops the terminal one, which is transcript order only on the plus
   strand; measured against the track it misplaces exactly one donor and one
   acceptor per minus-strand gene (MYBPC3 33/34, BRCA1 21/22, ABCC4 29/30) while
   plus-strand genes agree exactly.

Metrics
-------
For the base model and the meta model, over rows with a resolvable expectation:

``loss_type_accuracy``
    Fraction where the channel carrying the largest **loss** matches
    ``expected_type``. This is the position-level version of "what kind of
    splicing change".

``site_localisation@k``
    Fraction where the largest-loss position is within ``k`` bp of
    ``expected_site`` (k = 2, 10, 50).

``median_dist_to_site``
    Median |largest-loss position − expected_site|.

``median_peak_offset``
    Median |largest-|Δ| position − variant position|. Descriptive: it says
    whether a model's signal stays local to the variant or wanders.

Usage
-----
::

    python examples/variant_analysis/05_variant_positional_accuracy.py \\
        --checkpoint output/meta_layer/m2s_v4_cleanannot/best.pt \\
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
        --output-dir output/m4_benchmarks/positional_m2s_v4_cleanannot

    # Quick smoke run
    python examples/variant_analysis/05_variant_positional_accuracy.py \\
        --checkpoint output/meta_layer/m2s_v4_cleanannot/best.pt \\
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
        --max-variants 20 --output-dir /tmp/positional_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment  # noqa: E402

setup_example_environment()

log = logging.getLogger(__name__)

_DONOR, _ACCEPTOR = 0, 1
_CHANNEL_NAME = {_DONOR: "donor", _ACCEPTOR: "acceptor"}

#: Distances (bp) at which site localisation is reported.
_LOCALISATION_RADII = (2, 10, 50)

#: ``c.236+1G>A`` -> ``+1``; ``c.4994-2A>T`` -> ``-2``; ``c.1935A>T`` -> None.
#: The leading coordinate may carry ``*`` (3' UTR) or ``-`` (5' UTR), so the
#: offset is taken from the group that directly precedes the allele change.
_HGVS_OFFSET = re.compile(r"c\.[*-]?\d+([+-]\d+)?(?=[ACGT]|_|del|dup|ins)")


def expected_site_type(hgvs: str) -> str | None:
    """Splice-site type a variant's HGVS nomenclature implies it disrupts.

    Parameters
    ----------
    hgvs : str
        HGVS string, e.g. ``"NM_005957.5:c.236+1G>A"``.

    Returns
    -------
    str or None
        ``"donor"`` for a positive intronic offset, ``"acceptor"`` for a
        negative one, ``None`` when the variant is exonic or unparseable.

    Examples
    --------
    >>> expected_site_type("NM_005957.5:c.236+1G>A")
    'donor'
    >>> expected_site_type("NM_006015.6:c.4994-2A>T")
    'acceptor'
    >>> expected_site_type("NM_022552.5:c.1935A>T") is None
    True
    """
    m = _HGVS_OFFSET.search(hgvs or "")
    if m is None or m.group(1) is None:
        return None
    return "donor" if m.group(1).startswith("+") else "acceptor"


def load_site_track(path: Path) -> dict[tuple[str, str], np.ndarray]:
    """Annotated splice sites keyed by ``(gene_name, splice_type)``.

    Positions are returned sorted so the nearest-site lookup can bisect.
    """
    df = pl.read_parquet(path).select("gene_name", "splice_type", "position")
    out: dict[tuple[str, str], np.ndarray] = {}
    for (gene, stype), sub in df.group_by(["gene_name", "splice_type"]):
        out[(str(gene), str(stype))] = np.sort(sub["position"].to_numpy())
    return out


def nearest_site(sites: np.ndarray, position: int) -> int | None:
    """Annotated position closest to ``position``, or None when there are none."""
    if sites.size == 0:
        return None
    i = int(np.searchsorted(sites, position))
    cands = [sites[j] for j in (i - 1, i) if 0 <= j < sites.size]
    return int(min(cands, key=lambda p: abs(int(p) - position)))


def _peak_and_loss(delta: np.ndarray, window_start: int) -> dict[str, Any]:
    """Largest |Δ| and largest loss over the donor/acceptor channels.

    ``delta`` is ``[L, 3]`` in genomic coordinate order; the ``neither`` channel
    is excluded because only donor/acceptor changes are splice-altering.
    """
    dv = delta[:, :2]
    flat_peak = int(np.abs(dv).argmax())
    prow, pcol = divmod(flat_peak, 2)
    flat_loss = int(dv.argmin())  # most negative == largest loss
    lrow, lcol = divmod(flat_loss, 2)
    return {
        "peak_pos": window_start + prow,
        "peak_type": _CHANNEL_NAME[pcol],
        "peak_delta": float(dv[prow, pcol]),
        "loss_pos": window_start + lrow,
        "loss_type": _CHANNEL_NAME[lcol],
        "loss_delta": float(dv[lrow, lcol]),
    }


def score_variants(
    variants: list,
    checkpoint: Path,
    fasta_path: Path,
    site_index: dict[tuple[str, str], np.ndarray],
    gene_strands: dict[str, str],
    device: str = "cpu",
    use_multimodal: bool = False,
) -> list[dict]:
    """Score each variant and record where each model put its largest loss."""
    from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import (
        VariantRunner,
    )

    runner = VariantRunner(
        meta_checkpoint=checkpoint,
        fasta_path=fasta_path,
        base_model="openspliceai",
        device=device,
    )

    rows: list[dict] = []
    n_errors = 0
    t0 = time.time()

    for i, v in enumerate(variants):
        strand = gene_strands.get(v.gene, v.strand)
        exp_type = expected_site_type(v.hgvs)
        exp_site = (
            nearest_site(site_index.get((v.gene, exp_type), np.empty(0, dtype=int)), v.position)
            if exp_type else None
        )

        try:
            r = runner.run(
                v.chrom, v.position, v.ref_allele, v.alt_allele,
                gene=v.gene, strand=strand, use_multimodal=use_multimodal,
            )
        except Exception as e:  # noqa: BLE001 — one bad row must not end the run
            n_errors += 1
            log.warning("Error scoring %s %s:%d: %s", v.gene, v.chrom, v.position, e)
            continue

        row: dict[str, Any] = {
            "gene": v.gene,
            "chrom": v.chrom,
            "position": v.position,
            "hgvs": v.hgvs,
            "strand": strand,
            "effect_type": v.effect_type,
            "expected_type": exp_type,
            "expected_site": exp_site,
            "variant_to_site": abs(exp_site - v.position) if exp_site is not None else None,
        }
        for who, d in (("meta", r.delta), ("base", r.base_delta)):
            stats = _peak_and_loss(d, r.window_start)
            row.update({f"{who}_{k}": val for k, val in stats.items()})
            row[f"{who}_peak_offset"] = abs(stats["peak_pos"] - v.position)
            row[f"{who}_type_correct"] = (
                None if exp_type is None else stats["loss_type"] == exp_type
            )
            row[f"{who}_dist_to_site"] = (
                None if exp_site is None else abs(stats["loss_pos"] - exp_site)
            )
        rows.append(row)

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  Scored {i + 1}/{len(variants)} ({rate:.1f}/s)")

    runner.close()
    print(f"\n  Complete: {len(rows)} scored, {n_errors} errors, {time.time() - t0:.0f}s")
    return rows


def compute_metrics(rows: list[dict]) -> dict[str, Any]:
    """Aggregate the per-variant records into the reported metric block."""
    typed = [r for r in rows if r["expected_type"] is not None]
    sited = [r for r in typed if r["expected_site"] is not None]

    metrics: dict[str, Any] = {
        "n_scored": len(rows),
        "n_with_expected_type": len(typed),
        "n_with_expected_site": len(sited),
        "n_exonic_excluded": len(rows) - len(typed),
        "expected_type_counts": {
            t: sum(1 for r in typed if r["expected_type"] == t) for t in ("donor", "acceptor")
        },
        "median_variant_to_site": (
            float(np.median([r["variant_to_site"] for r in sited])) if sited else None
        ),
    }

    # MutSpliceDB is dominated by canonical splice-site variants, so for most
    # rows the variant sits 1 bp from the site it disrupts and localisation is
    # satisfied by any locally-correct call. Stratifying by that distance
    # separates the trivial regime from the one that actually asks the model to
    # place the site: `variant_to_site` is annotation-derived, never predicted.
    strata = {
        "canonical (<=2 bp)": [r for r in sited if r["variant_to_site"] <= 2],
        "extended (3-10 bp)": [r for r in sited if 2 < r["variant_to_site"] <= 10],
        "distal (>10 bp)": [r for r in sited if r["variant_to_site"] > 10],
    }
    metrics["strata_counts"] = {k: len(v) for k, v in strata.items()}

    for who in ("base", "meta"):
        block: dict[str, Any] = {}
        if typed:
            block["loss_type_accuracy"] = float(
                np.mean([r[f"{who}_type_correct"] for r in typed])
            )
            # Reported separately because the two are not equally hard: on
            # ClinVar the meta layer loses ~20 points on acceptors specifically.
            block["loss_type_accuracy_by_type"] = {
                t: float(np.mean([r[f"{who}_type_correct"]
                                  for r in typed if r["expected_type"] == t]))
                for t in ("donor", "acceptor")
                if any(r["expected_type"] == t for r in typed)
            }
        if sited:
            d = np.array([r[f"{who}_dist_to_site"] for r in sited], dtype=float)
            block["median_dist_to_site"] = float(np.median(d))
            for k in _LOCALISATION_RADII:
                block[f"site_localisation@{k}"] = float((d <= k).mean())
            block["site_localisation@2_by_stratum"] = {
                name: float(np.mean([r[f"{who}_dist_to_site"] <= 2 for r in sub]))
                for name, sub in strata.items() if sub
            }
        if rows:
            block["median_peak_offset"] = float(
                np.median([r[f"{who}_peak_offset"] for r in rows])
            )
        metrics[who] = block

    return metrics


def write_manifest(out_dir: Path, checkpoint: Path, metrics: dict, arch: str) -> None:
    """Write the MANIFEST.yaml every ``output/`` artifact in this project carries."""
    meta_acc = metrics["meta"].get("loss_type_accuracy")
    base_acc = metrics["base"].get("loss_type_accuracy")
    note = (
        f"Position-level variant eval on MutSpliceDB (n={metrics['n_scored']}). "
        f"Largest-loss site-type accuracy: base {base_acc:.3f}, meta {meta_acc:.3f}. "
        "Truth = HGVS intronic offset (donor/acceptor) + nearest MANE site; "
        "MutSpliceDB carries no measured induced-site coordinate."
    )
    lines = [
        "status: baseline",
        "produced_by:",
        "- examples/variant_analysis/05_variant_positional_accuracy.py",
        "superseded_by: null",
        f"notes: {json.dumps(note)}",
        "tags:",
        f"- arch:{arch}",
        "- variant_effect",
        "- position_level",
        "referenced_by:",
        "- examples/variant_analysis/results/m4_variant_arm_status.md",
    ]
    (out_dir / "MANIFEST.yaml").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Position-level variant delta evaluation on MutSpliceDB",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--mutsplicedb", type=Path,
                        default=Path("data/mutsplicedb/splice_sites_induced.tsv"))
    parser.add_argument("--site-track", type=Path,
                        default=Path("data/mane/GRCh38/splice_sites_track.parquet"),
                        help="Strand-resolved MANE splice-site track used as truth.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--multimodal", action="store_true",
                        help="Extract dense features. Off by default: for an SNV they are "
                             "identical between ref and alt and cancel in the subtraction.")
    parser.add_argument("--gffutils-db", type=Path,
                        default=Path("data/mane/GRCh38/annotations.db"))
    parser.add_argument("--resolve-hgvs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("=" * 62)
    print("Position-level variant evaluation (MutSpliceDB)")
    print("=" * 62)

    from agentic_spliceai.splice_engine.meta_layer.data.mutsplicedb_loader import (
        MutSpliceDBLoader,
    )
    from agentic_spliceai.splice_engine.resources import get_model_resources

    resolver = None
    if args.resolve_hgvs:
        from agentic_spliceai.splice_engine.utils.hgvs_resolver import HgvsResolver
        if not args.gffutils_db.exists():
            raise FileNotFoundError(
                f"gffutils DB not found at {args.gffutils_db}; pass --no-resolve-hgvs."
            )
        resolver = HgvsResolver(args.gffutils_db)
        print(f"  HGVS resolver: enabled (db={args.gffutils_db.name})")

    variants = list(MutSpliceDBLoader(args.mutsplicedb, resolver=resolver).iter_variants())
    if args.max_variants:
        variants = variants[:args.max_variants]
    print(f"  Variants:      {len(variants)}")

    site_index = load_site_track(args.site_track)
    print(f"  Site track:    {args.site_track.name} "
          f"({sum(v.size for v in site_index.values()):,} sites)")

    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    gtf_path = str(get_model_resources("openspliceai").get_gtf_path())
    ann = extract_gene_annotations(gtf_path, verbosity=0)
    gene_strands = {
        r["gene_name"]: r.get("strand", "+")
        for r in ann.iter_rows(named=True) if r.get("gene_name")
    }

    print(f"  Checkpoint:    {args.checkpoint.parent.name}/{args.checkpoint.name}\n")
    rows = score_variants(
        variants, args.checkpoint, args.fasta, site_index, gene_strands,
        device=args.device, use_multimodal=args.multimodal,
    )
    if not rows:
        print("  ERROR: no variants scored")
        return 1

    metrics = compute_metrics(rows)

    print(f"\n{'=' * 62}")
    print("  Position-level results")
    print(f"{'=' * 62}")
    print(f"  Scored:                    {metrics['n_scored']}")
    print(f"  With an expected site:     {metrics['n_with_expected_site']} "
          f"(donor {metrics['expected_type_counts']['donor']}, "
          f"acceptor {metrics['expected_type_counts']['acceptor']}; "
          f"{metrics['n_exonic_excluded']} exonic excluded)")
    print(f"  Variant to its site:       median {metrics['median_variant_to_site']:.0f} bp\n")
    print(f"  {'metric':<26} {'base':>9} {'meta':>9}")
    for key, label in (
        ("loss_type_accuracy", "largest-loss type acc"),
        ("site_localisation@2", "on site (±2 bp)"),
        ("site_localisation@10", "within ±10 bp"),
        ("site_localisation@50", "within ±50 bp"),
        ("median_dist_to_site", "median dist to site"),
        ("median_peak_offset", "median peak offset"),
    ):
        b, m = metrics["base"].get(key), metrics["meta"].get(key)
        if b is None or m is None:
            continue
        print(f"  {label:<26} {b:>9.3f} {m:>9.3f}")

    print(f"\n  {'largest-loss type acc':<26} {'base':>9} {'meta':>9}")
    for t in ("donor", "acceptor"):
        b = metrics["base"].get("loss_type_accuracy_by_type", {}).get(t)
        m = metrics["meta"].get("loss_type_accuracy_by_type", {}).get(t)
        if b is None or m is None:
            continue
        n = metrics["expected_type_counts"][t]
        print(f"    {t + f' (n={n})':<24} {b:>9.3f} {m:>9.3f}")

    print(f"\n  {'on site (±2 bp), by how far':<26} {'base':>9} {'meta':>9}")
    for name, n in metrics["strata_counts"].items():
        b = metrics["base"].get("site_localisation@2_by_stratum", {}).get(name)
        m = metrics["meta"].get("site_localisation@2_by_stratum", {}).get(name)
        if b is None or m is None:
            continue
        print(f"    {name + f' n={n}':<24} {b:>9.3f} {m:>9.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval": "variant_positional_accuracy",
        "corpus": "mutsplicedb",
        "checkpoint": str(args.checkpoint),
        "site_track": str(args.site_track),
        "use_multimodal": args.multimodal,
        "truth_definition": {
            "expected_type": "sign of the HGVS intronic offset (+ -> donor, - -> acceptor)",
            "expected_site": "nearest annotated MANE site of expected_type in the same gene",
            "note": "MutSpliceDB carries no measured induced-site coordinate; its "
                    "`position` is the variant.",
        },
        "metrics": metrics,
    }
    (args.output_dir / "eval_results.json").write_text(json.dumps(payload, indent=2))
    pl.DataFrame(rows, strict=False).write_parquet(args.output_dir / "per_variant.parquet")

    cfg_path = args.checkpoint.parent / "config.pt"
    arch = "concat_fusion" if cfg_path.exists() else "unknown"
    write_manifest(args.output_dir, args.checkpoint, metrics, arch)

    print(f"\n  Wrote {args.output_dir}/eval_results.json, per_variant.parquet, MANIFEST.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

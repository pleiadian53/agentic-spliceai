#!/usr/bin/env python
"""M3 Phase D / Tier 0 — honest evaluation of the novel-splice-site recognizer.

Replaces the circular in-distribution validation number (donor PR-AUC ~0.30 on
a holdout of the same SpliceVault-dominated label pool) with an **anti-circular,
use-case-shaped** evaluation:

* **Per-gene precision@k / recall@k** (not genome-wide PR-AUC) — M3's real use is
  "for this locus, give me the top candidate unannotated sites to inspect".
* **Independent truth**: ENCODE long-read novel junctions (D1) + held-out disease
  anchors (D2), neither of which is the SpliceVault training signal.
* **Novelty post-filter**: annotated sites are set-subtracted *before* ranking, so
  precision@k measures novelty, not "is this a canonical site".
* **Four (+1) models on identical candidates**: base / OpenSpliceAI (does
  multimodal M3 beat sequence-only?), M1-S (zero-shot transfer), M2-S (the
  comparator to beat), M3 v1, and M3 v1 with multimodal channels zeroed (does the
  multimodal evidence buy anything at all?).

Eval universe = SpliceAI test chromosomes (1,3,5,7,9), which are held out for
M1-S / M2-S / M3-v1 alike → D1 is clean without any extra in-training split.

Three modes (the cache build is the only bigWig/pod-dependent step):

    # 1. LOCAL — resolve the truth-containing test-gene universe
    python 13_evaluate_m3_novel.py --mode emit-universe \\
        --output-dir output/meta_layer/m3_eval_d1

    # 2. POD (A40, network volume) — build the shared 9-channel .npz cache
    python 13_evaluate_m3_novel.py --mode build-cache \\
        --gene-list output/meta_layer/m3_eval_d1/eval_genes.txt \\
        --cache-dir output/meta_layer/m3_eval_d1/gene_cache \\
        --bigwig-cache /runpod-volume/bigwig_cache --device cuda

    # 3. LOCAL — score all models from the rsynced cache, report metrics
    python 13_evaluate_m3_novel.py --mode eval \\
        --cache-dir output/meta_layer/m3_eval_d1/gene_cache \\
        --universe output/meta_layer/m3_eval_d1/eval_genes.parquet \\
        --output-dir output/meta_layer/m3_eval_d1
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment  # noqa: E402
setup_example_environment()

# Local eval library (pure metric functions + gene-universe resolver).
sys.path.insert(0, str(Path(__file__).parent))
import _m3_novel_eval as m3eval  # noqa: E402

log = logging.getLogger(__name__)

# ── Default data / model locations ─────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]  # examples/meta_layer/<file> -> repo root
D1_TRUTH = REPO / "data/encode_longread/GRCh38/longread_truth_novel.parquet"
D2_ANCHORS = REPO / "data/mane/GRCh38/m3_labels/disease_anchors.parquet"
ANNOTATION_MASK = REPO / "data/mane/GRCh38/m3_labels/annotation_mask.parquet"

DEFAULT_MODELS = [
    # (display name, model dir relative to repo, mm-transform tag)
    ("base",       None,                                  "base"),
    ("M1-S",       "output/meta_layer/m1s_v4_cleanannot", "full9"),
    ("M2-S",       "output/meta_layer/m2s_v4_cleanannot", "full9"),
    ("M3-v1",      "output/meta_layer/m3_v1",             "m3"),
    ("M3-v1-mm0",  "output/meta_layer/m3_v1",             "m3_zero"),
    # Tier 1: confirmed-only retrain (skipped automatically until the dir exists locally)
    ("M3-v1.1",    "output/meta_layer/m3s_v1_1_confirmed", "m3"),
    # Disease-anchor fold: SF3B1/ENCODE-KD positivized (w=5), TDP-43 masked.
    # Tests SF3B1 transfer on D2, anti-circular by chromosome. Skipped until built.
    ("M3-anchor",  "output/meta_layer/m3s_anchorpos",      "m3"),
]

KS = (5, 10, 20)
MAX_K = max(KS)


# ---------------------------------------------------------------------------
# Gene-annotation helpers (shared coordinate frame with build_gene_cache)
# ---------------------------------------------------------------------------

def _resource_gene_annotations():
    """MANE gene annotations via the same path build_gene_cache uses on the pod."""
    from agentic_spliceai.splice_engine.resources import get_model_resources
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    resources = get_model_resources("openspliceai")
    reg = resources.get_registry()
    gene_annotations = extract_gene_annotations(str(reg.get_gtf_path()), verbosity=0)
    return resources, reg, gene_annotations


def _build_interval_lut(gene_annotations) -> Dict[str, m3eval.GeneInterval]:
    """gene_id AND gene_name -> GeneInterval (bare chrom, 0-based start)."""
    lut: Dict[str, m3eval.GeneInterval] = {}
    for row in gene_annotations.iter_rows(named=True):
        chrom = m3eval.strip_chr(str(row.get("chrom") or row.get("seqname")))
        gi = m3eval.GeneInterval(
            gene_id=str(row["gene_id"]), chrom=chrom,
            start=int(row["start"]), end=int(row["end"]), strand=str(row["strand"]),
        )
        lut[str(row["gene_id"])] = gi
        if row.get("gene_name"):
            lut[str(row["gene_name"])] = gi
    return lut


# ---------------------------------------------------------------------------
# Mode: emit-universe (local)
# ---------------------------------------------------------------------------

def mode_emit_universe(args) -> int:
    import polars as pl
    from agentic_spliceai.splice_engine.eval.splitting import (
        build_gene_split, gene_chromosomes_from_dataframe,
    )

    resources, reg, gene_annotations = _resource_gene_annotations()
    gene_chroms = gene_chromosomes_from_dataframe(gene_annotations)

    # Paralog-clean SpliceAI test set (same guard as 08/09).
    gene_seqs = None
    if args.remove_paralogs:
        try:
            from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
                extract_gene_sequences,
            )
            log.info("Extracting gene sequences for paralog-clean test set "
                     "(%d genes)...", gene_annotations.height)
            gene_seqs = extract_gene_sequences(
                gene_annotations, str(resources.get_fasta_path())
            )
        except Exception as e:  # mappy missing / FASTA issue — test chroms already held out
            log.warning("Paralog removal skipped (%s); test chroms are still held out.", e)

    split = build_gene_split(
        gene_chroms, preset="spliceai", val_fraction=0.0, gene_sequences=gene_seqs,
    )
    test_genes = sorted(split.test_genes)
    test_chroms_bare = sorted({m3eval.strip_chr(gene_chroms[g]) for g in test_genes
                               if g in gene_chroms})
    log.info("SpliceAI test set: %d genes on chroms %s (%d paralogs removed)",
             len(test_genes), test_chroms_bare, len(split.test_paralogs_removed))

    # Intervals for the test genes.
    lut = _build_interval_lut(gene_annotations)
    intervals: List[m3eval.GeneInterval] = []
    missing = 0
    for g in test_genes:
        gi = lut.get(g)
        if gi is None:
            missing += 1
            continue
        intervals.append(m3eval.GeneInterval(g, gi.chrom, gi.start, gi.end, gi.strand))
    if missing:
        log.warning("%d test genes had no interval in annotations (skipped)", missing)

    # Truth sets restricted to the test chroms.
    d1 = m3eval.load_sites_parquet(
        args.d1_truth, keep_chroms_bare=test_chroms_bare,
        min_biosamples=1, extra_cols=("n_biosamples",),
    )
    d2 = m3eval.load_sites_parquet(
        args.d2_anchors, keep_chroms_bare=test_chroms_bare, is_novel_only=True,
    )
    log.info("Truth on test chroms: D1=%d sites, D2(novel)=%d sites", d1.height, d2.height)

    kept = m3eval.resolve_truth_genes(intervals, [d1, d2])
    log.info("Truth-containing test genes: %d / %d", len(kept), len(intervals))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / "eval_genes.txt"
    ids_path.write_text("\n".join(g.gene_id for g in kept) + "\n")
    pl.DataFrame({
        "gene_id": [g.gene_id for g in kept],
        "chrom": [g.chrom for g in kept],
        "start": [g.start for g in kept],
        "end": [g.end for g in kept],
        "strand": [g.strand for g in kept],
    }).write_parquet(out_dir / "eval_genes.parquet")

    manifest = {
        "mode": "emit-universe",
        "test_chroms": test_chroms_bare,
        "n_test_genes": len(intervals),
        "n_truth_genes": len(kept),
        "d1_sites_on_test_chroms": d1.height,
        "d2_novel_sites_on_test_chroms": d2.height,
        "remove_paralogs": bool(gene_seqs is not None),
        "d1_truth": str(args.d1_truth),
        "d2_anchors": str(args.d2_anchors),
    }
    (out_dir / "universe_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(kept)} truth-containing genes -> {ids_path}")
    print(f"Manifest: {out_dir / 'universe_manifest.json'}")
    return 0


# ---------------------------------------------------------------------------
# Mode: build-cache (pod)
# ---------------------------------------------------------------------------

def mode_build_cache(args) -> int:
    import pandas as pd
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        build_gene_cache,
    )
    from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
        DenseFeatureExtractor, DenseFeatureConfig,
    )
    from agentic_spliceai.splice_engine.eval.streaming_metrics import preflight_check

    gene_ids = [ln.strip() for ln in Path(args.gene_list).read_text().splitlines()
                if ln.strip()]
    log.info("Building cache for %d genes", len(gene_ids))

    resources, reg, gene_annotations = _resource_gene_annotations()
    fasta_path = str(resources.get_fasta_path())
    splice_sites_path = Path(reg.stash) / "splice_sites_enhanced.tsv"
    base_scores_dir = (
        args.base_scores_dir
        or reg.get_base_model_eval_dir("openspliceai") / "precomputed"
    )
    if not splice_sites_path.exists():
        print(f"ERROR: splice sites not found at {splice_sites_path}")
        return 1
    preflight_check(needs_bigwig=True, needs_pyfaidx=True,
                    fasta_path=fasta_path, base_scores_dir=base_scores_dir)

    # Hard preflight: base scores MUST exist for every chrom in the gene list.
    # Missing files silently fall back to a uniform 1/3 prior (see
    # _load_chrom_base_scores), which would invalidate BOTH the base baseline
    # and every meta model's base-score blend — a silent, fatal error.
    lut = _build_interval_lut(gene_annotations)
    chroms = sorted({lut[g].chrom for g in gene_ids if g in lut})
    missing = [
        ch for ch in chroms
        if not (base_scores_dir / f"predictions_chr{ch}.parquet").exists()
        and not (base_scores_dir / f"predictions_{ch}.parquet").exists()
    ]
    if missing:
        print(f"ERROR: base-score predictions missing for chroms {missing} in "
              f"{base_scores_dir}.\n  build_gene_cache would silently use a uniform "
              f"1/3 prior and invalidate the eval. Stage predictions_chr*.parquet first.")
        return 1
    print(f"Base scores present for chroms {chroms}")

    splice_sites_df = pd.read_csv(splice_sites_path, sep="\t")
    extractor = DenseFeatureExtractor(
        DenseFeatureConfig(build="GRCh38", bigwig_cache_dir=args.bigwig_cache)
    )
    # Full 9-channel cache shared by all models (M3 slices out junction at eval).
    assert extractor.num_channels == 9, (
        f"Expected 9 mm channels, got {extractor.num_channels}: {extractor.channel_names}"
    )

    cache_dir = Path(args.cache_dir)
    t0 = time.time()
    build_gene_cache(
        gene_ids, splice_sites_df, fasta_path, base_scores_dir,
        extractor, gene_annotations, cache_dir=cache_dir,
    )
    extractor.close()
    n_npz = len(list(cache_dir.glob("*.npz")))
    print(f"\nCache built: {n_npz} .npz in {cache_dir} ({time.time() - t0:.1f}s)")
    return 0


# ---------------------------------------------------------------------------
# Mode: eval (local)
# ---------------------------------------------------------------------------

def _mm_transform(tag: str, m3_keep: List[int]):
    if tag == "full9":
        return lambda mm: mm
    if tag == "m3":
        return lambda mm: mm[:, m3_keep]
    if tag == "m3_zero":
        return lambda mm: np.zeros((mm.shape[0], len(m3_keep)), dtype=np.float32)
    raise ValueError(tag)


def _load_models(model_specs, device):
    """Return {name: (model_or_None, cfg_or_None, mm_transform)}."""
    from agentic_spliceai.splice_engine.meta_layer.models.loader import load_meta_model
    from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
        CHANNEL_NAMES, M3_EXCLUDE,
    )
    m3_keep = [i for i, n in enumerate(CHANNEL_NAMES) if n not in M3_EXCLUDE]

    models = {}
    for name, rel_dir, tag in model_specs:
        if tag == "base":
            models[name] = (None, None, None)
            continue
        mdir = REPO / rel_dir
        if not (mdir / "config.pt").exists() or not (mdir / "best.pt").exists():
            log.warning("Model %s not found at %s — skipping", name, mdir)
            continue
        model, cfg = load_meta_model(mdir, device)
        transform = _mm_transform(tag, m3_keep)
        # sanity: transformed mm width must equal the model's expected mm_channels
        want = cfg.mm_channels
        got = len(m3_keep) if tag.startswith("m3") else len(CHANNEL_NAMES)
        if want != got:
            log.warning("Model %s expects mm_channels=%d but transform yields %d",
                        name, want, got)
        models[name] = (model, cfg, transform)
        log.info("Loaded %s (%s): mm_channels=%d", name, cfg.variant, cfg.mm_channels)
    return models


def mode_eval(args) -> int:
    import polars as pl
    from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz,
    )
    from agentic_spliceai.splice_engine.utils.device import resolve_device

    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Gene universe (intervals) ────────────────────────────────────────
    uni = pl.read_parquet(args.universe)
    genes = [
        m3eval.GeneInterval(r["gene_id"], r["chrom"], int(r["start"]),
                            int(r["end"]), r["strand"])
        for r in uni.iter_rows(named=True)
    ]
    test_chroms_bare = sorted({g.chrom for g in genes})
    log.info("Eval universe: %d genes on chroms %s", len(genes), test_chroms_bare)

    # ── Truth + annotation indices ──────────────────────────────────────
    annotation = m3eval.SiteIndex.from_df(
        m3eval.load_sites_parquet(args.annotation, keep_chroms_bare=test_chroms_bare)
    )
    truth_sets = {
        "D1": m3eval.SiteIndex.from_df(m3eval.load_sites_parquet(
            args.d1_truth, keep_chroms_bare=test_chroms_bare)),
        "D1_hiconf": m3eval.SiteIndex.from_df(m3eval.load_sites_parquet(
            args.d1_truth, keep_chroms_bare=test_chroms_bare, min_biosamples=2)),
        "D2": m3eval.SiteIndex.from_df(m3eval.load_sites_parquet(
            args.d2_anchors, keep_chroms_bare=test_chroms_bare, is_novel_only=True)),
    }

    # ── Models ──────────────────────────────────────────────────────────
    models = _load_models(DEFAULT_MODELS, device)
    if not models:
        print("ERROR: no models loaded")
        return 1

    # accs[truthset][model][splice_type]
    accs = {
        ts: {m: {"donor": m3eval.KMetricAccumulator(KS),
                 "acceptor": m3eval.KMetricAccumulator(KS)}
             for m in models}
        for ts in truth_sets
    }

    # ── Per-gene scoring loop (one .npz in memory at a time) ─────────────
    t0 = time.time()
    n_scored = n_skipped = 0
    for i, gene in enumerate(genes):
        npz = cache_dir / f"{gene.gene_id}.npz"
        if not npz.exists():
            n_skipped += 1
            continue
        data = _load_gene_npz(npz)
        L = len(data["sequence"])
        if L != (gene.end - gene.start):
            # Cache length must equal the annotated window, or index->position breaks.
            log.warning("Gene %s: cache length %d != interval %d — skipping",
                        gene.gene_id, L, gene.end - gene.start)
            n_skipped += 1
            del data
            continue

        base = data["base_scores"]
        mm9 = data["mm_features"]
        for name, (model, cfg, transform) in models.items():
            if model is None:
                probs = base
            else:
                gd = {"sequence": data["sequence"], "base_scores": base,
                      "mm_features": transform(mm9)}
                probs = infer_full_gene(
                    model, gd, context_padding=cfg.effective_context_padding,
                    device=device,
                )
            for ts, idx in truth_sets.items():
                m3eval.evaluate_gene(probs, gene, annotation, idx, accs[ts][name], MAX_K)
        n_scored += 1
        del data, base, mm9
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(genes)} genes ({time.time()-t0:.0f}s)...")

    print(f"Scored {n_scored} genes ({n_skipped} skipped) in {time.time()-t0:.1f}s")
    if n_scored == 0:
        print("ERROR: no genes scored — check --cache-dir")
        return 1

    # ── Aggregate + report ──────────────────────────────────────────────
    results: Dict[str, Dict[str, dict]] = {}
    for ts in truth_sets:
        per_model = {}
        for name in models:
            d = accs[ts][name]["donor"].result()
            a = accs[ts][name]["acceptor"].result()
            per_model[name] = {
                "donor": d, "acceptor": a,
                "combined": m3eval.combine_type_results(d, a, KS),
            }
        results[ts] = per_model
        print(f"\n{'='*74}\n{ts}: per-gene precision@k / recall@k "
              f"(macro over truth-containing genes)\n{'='*74}")
        print(m3eval.format_comparison_table(per_model, KS))

    payload = {
        "mode": "eval", "device": str(device), "ks": list(KS),
        "n_genes_scored": n_scored, "n_genes_skipped": n_skipped,
        "test_chroms": test_chroms_bare,
        "models": [n for n in models],
        "results": results,
    }
    (out_dir / "m3_eval_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    _write_markdown(out_dir / "m3_eval_summary.md", results, KS)
    print(f"\nResults -> {out_dir / 'm3_eval_metrics.json'}")
    print(f"Summary -> {out_dir / 'm3_eval_summary.md'}")
    return 0


def _write_markdown(path: Path, results, ks):
    lines = ["# M3 Phase D / Tier 0 — evaluation results\n",
             "Per-gene precision@k / recall@k (macro over truth-containing genes), "
             "donor+acceptor combined. Novelty post-filtered against the annotation "
             "union before ranking.\n"]
    for ts, per_model in results.items():
        n_genes = next(iter(per_model.values()))["combined"]["n_genes_with_truth"]
        n_truth = next(iter(per_model.values()))["combined"]["n_truth_sites"]
        lines.append(f"\n## {ts}  (genes with truth: {n_genes}, truth sites: {n_truth})\n")
        head = "| model | " + " | ".join(f"P@{k} | R@{k}" for k in ks) + " |"
        sep = "|" + "---|" * (1 + 2 * len(ks))
        lines += [head, sep]
        for name, res in per_model.items():
            c = res["combined"]
            cells = " | ".join(
                f"{c['macro_precision_at_k'][k]:.3f} | {c['macro_recall_at_k'][k]:.3f}"
                for k in ks
            )
            lines.append(f"| {name} | {cells} |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["emit-universe", "build-cache", "eval"],
                   default="eval")
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "output/meta_layer/m3_eval_d1")
    p.add_argument("--gene-list", type=Path, default=None,
                   help="build-cache: eval_genes.txt (one gene id per line)")
    p.add_argument("--universe", type=Path, default=None,
                   help="eval: eval_genes.parquet (gene intervals)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="build-cache/eval: dir of per-gene .npz")
    p.add_argument("--bigwig-cache", type=Path, default=None,
                   help="build-cache: local bigWig cache dir (pod: /runpod-volume/bigwig_cache)")
    p.add_argument("--base-scores-dir", type=Path, default=None)
    p.add_argument("--d1-truth", type=Path, default=D1_TRUTH)
    p.add_argument("--d2-anchors", type=Path, default=D2_ANCHORS)
    p.add_argument("--annotation", type=Path, default=ANNOTATION_MASK)
    p.add_argument("--remove-paralogs", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--device", default="cpu",
                   help="eval/build-cache device (default cpu for local eval; "
                        "pass cuda on the pod)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.mode == "emit-universe":
        return mode_emit_universe(args)
    if args.mode == "build-cache":
        for req in ("gene_list", "cache_dir"):
            if getattr(args, req) is None:
                print(f"ERROR: --{req.replace('_','-')} required for build-cache")
                return 1
        return mode_build_cache(args)
    # eval
    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "gene_cache"
    if args.universe is None:
        args.universe = args.output_dir / "eval_genes.parquet"
    return mode_eval(args)


if __name__ == "__main__":
    sys.exit(main())

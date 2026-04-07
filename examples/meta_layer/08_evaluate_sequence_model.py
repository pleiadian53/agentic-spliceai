#!/usr/bin/env python
"""Evaluate a trained M*-S meta-splice model against base model scores.

Compares the meta-layer's refined [L, 3] predictions against the raw
base model scores on held-out test genes, answering: "Does the
meta-layer actually improve over the base model?"

Uses streaming evaluation — only one gene's arrays in memory at a time,
bounded to ~5 MB regardless of total gene count.

Metrics reported:
- Per-class and macro PR-AUC
- Accuracy, top-1, top-2
- Per-class precision, recall, F1
- FN/FP counts and reduction vs base model
- SpliceAI paper top-k accuracy (k = 0.5, 1, 2, 4 × n_true)

Usage:
    # Build test cache + evaluate M1-S (chr1,3,5,7,9)
    python 08_evaluate_sequence_model.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --build-cache

    # Evaluate from existing cache
    python 08_evaluate_sequence_model.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --cache-dir output/meta_layer/gene_cache/test

    # Evaluate on specific chromosomes
    python 08_evaluate_sequence_model.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --test-chroms chr1 chr3 \\
        --build-cache

    # Pure FASTA inference (no annotations, no features)
    python 08_evaluate_sequence_model.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --fasta /path/to/sequences.fa
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

log = logging.getLogger(__name__)


from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene


# ---------------------------------------------------------------------------
# FASTA inference (Case 4)
# ---------------------------------------------------------------------------


def _run_fasta_inference(
    model,
    cfg,
    args: argparse.Namespace,
    device,
) -> int:
    """Run pure inference on arbitrary FASTA sequences.

    No gene annotations, no base model scores, no multimodal features.
    Uses uniform 1/3 base-score prior and zero multimodal features.
    Outputs per-position [donor, acceptor, neither] probabilities.
    """
    import pyfaidx

    fasta_path = args.fasta
    if not fasta_path.exists():
        print(f"ERROR: FASTA not found: {fasta_path}")
        return 1

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mm_channels = cfg.mm_channels
    min_recommended = 5001 + 400  # window + context

    print(f"\n{'='*70}")
    print("FASTA Inference Mode")
    print(f"{'='*70}")
    print(f"  Input:  {fasta_path}")
    print(f"  Format: {args.output_format}")

    fasta = pyfaidx.Fasta(str(fasta_path))
    seq_names = list(fasta.keys())
    print(f"  Sequences: {len(seq_names)}")

    rows: List[Tuple[str, int, float, float, float]] = []
    t0 = time.time()

    for i, name in enumerate(seq_names):
        sequence = str(fasta[name][:]).upper()
        seq_len = len(sequence)

        if seq_len < 100:
            log.warning("Skipping %s: too short (%d bp)", name, seq_len)
            continue
        if seq_len < min_recommended:
            log.warning(
                "%s: %d bp (< %d recommended) — predictions may be degraded",
                name, seq_len, min_recommended,
            )

        gene_data = {
            "sequence": sequence,
            "base_scores": np.full((seq_len, 3), 1.0 / 3, dtype=np.float32),
            "mm_features": np.zeros((seq_len, mm_channels), dtype=np.float32),
        }

        probs = infer_full_gene(model, gene_data, device=device)  # [L, 3]

        for pos in range(seq_len):
            rows.append((
                name,
                pos,
                float(probs[pos, 0]),
                float(probs[pos, 1]),
                float(probs[pos, 2]),
            ))

        if (i + 1) % 10 == 0 or (i + 1) == len(seq_names):
            print(f"  Processed {i+1}/{len(seq_names)} sequences...")

    elapsed = time.time() - t0
    total_positions = len(rows)
    print(f"  Done: {total_positions:,} positions in {elapsed:.1f}s")

    # ── Write output ────────────────────────────────────────────────
    if args.output_format == "parquet":
        import polars as pl

        df = pl.DataFrame(
            {
                "sequence_id": [r[0] for r in rows],
                "position": [r[1] for r in rows],
                "donor_prob": [r[2] for r in rows],
                "acceptor_prob": [r[3] for r in rows],
                "neither_prob": [r[4] for r in rows],
            },
        )
        out_path = output_dir / "fasta_predictions.parquet"
        df.write_parquet(out_path)
    else:
        out_path = output_dir / "fasta_predictions.tsv"
        with open(out_path, "w") as f:
            f.write("sequence_id\tposition\tdonor_prob\tacceptor_prob\tneither_prob\n")
            for seq_id, pos, dp, ap, np_ in rows:
                f.write(f"{seq_id}\t{pos}\t{dp:.6f}\t{ap:.6f}\t{np_:.6f}\n")

    print(f"\n  Output: {out_path}")
    print(f"{'='*70}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate M*-S meta-splice model vs base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to trained model checkpoint (best.pt)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config.pt (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory with gene cache (.npz files). "
             "Default: <checkpoint-dir>/gene_cache/test",
    )
    parser.add_argument(
        "--build-cache", action="store_true",
        help="Build gene cache for test genes before evaluating. "
             "Requires FASTA, base scores, and feature data.",
    )
    parser.add_argument(
        "--base-scores-dir", type=Path, default=None,
        help="Override base model predictions directory for cache building.",
    )
    parser.add_argument(
        "--bigwig-cache", type=Path, default=None,
        help="Local directory with cached conservation bigWig files.",
    )
    parser.add_argument(
        "--test-chroms", nargs="+", default=None,
        help="Evaluate on genes from these chromosomes "
             "(default: SpliceAI test set chr1,3,5,7,9)",
    )
    parser.add_argument(
        "--genes", nargs="+", default=None,
        help="Evaluate on specific genes by name/ID (inference mode). "
             "Overrides --test-chroms.",
    )
    parser.add_argument(
        "--annotation-source", choices=["mane", "ensembl"], default="mane",
        help="Annotation source for gene annotations (default: mane)",
    )
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Limit number of genes for quick testing",
    )
    parser.add_argument(
        "--sweep-thresholds", action="store_true",
        help="Sweep classification thresholds to find optimal precision-recall "
             "operating point.  Reports F1-optimal threshold and best precision "
             "at >= 95%% recall.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Apply temperature scaling with a fixed T value.  "
             "Use --calibrate-temperature to learn T from validation genes.",
    )
    parser.add_argument(
        "--calibrate-temperature", action="store_true",
        help="Learn optimal temperature T from validation set genes "
             "(SpliceAI val: chr2,4,6,8,10) before test evaluation.  "
             "Requires gene cache for validation chromosomes.",
    )
    parser.add_argument(
        "--val-cache-dir", type=Path, default=None,
        help="Gene cache for validation set (used with --calibrate-temperature). "
             "Default: <checkpoint-dir>/gene_cache/val",
    )
    parser.add_argument(
        "--zero-channels", nargs="+", default=None,
        help="Ablation: zero out specific multimodal channels before inference. "
             "Channel names: phylop_score, phastcons_score, h3k36me3_max, "
             "h3k4me3_max, atac_max, dnase_max, junction_log1p, "
             "junction_has_support, rbp_n_bound.  Use 'all' to zero all channels "
             "(sequence + base scores only).  Reuses existing cache.",
    )
    parser.add_argument(
        "--fasta", type=Path, default=None,
        help="FASTA file for pure inference mode.  Runs sliding-window "
             "prediction on each sequence using uniform base-score prior "
             "and zero multimodal features.  Mutually exclusive with "
             "--cache-dir / --build-cache / --genes / --test-chroms.",
    )
    parser.add_argument(
        "--output-format", choices=["tsv", "parquet"], default="tsv",
        help="Output format for FASTA inference (default: tsv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for evaluation results (default: checkpoint dir)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for inference (default: cpu)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    # ── Load model ───────────────────────────────────────────────────
    config_path = args.config or args.checkpoint.parent / "config.pt"
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}")
        return 1

    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceModel, MetaSpliceConfig,
    )

    device = torch.device(args.device)
    torch.serialization.add_safe_globals([MetaSpliceConfig])
    cfg = torch.load(config_path, map_location="cpu", weights_only=True)
    model = MetaSpliceModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.variant}, {n_params:,} params")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")

    # ── FASTA inference mode (Case 4) ────────────────────────────────
    if args.fasta:
        return _run_fasta_inference(model, cfg, args, device)

    # ── Resolve test genes ───────────────────────────────────────────
    from agentic_spliceai.splice_engine.resources import get_model_resources, get_genomic_registry
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import extract_gene_annotations
    from agentic_spliceai.splice_engine.eval.splitting import (
        build_gene_split, gene_chromosomes_from_dataframe,
    )

    resources = get_model_resources("openspliceai")
    ann_src = args.annotation_source
    if ann_src == "ensembl":
        ann_registry = get_genomic_registry(build="GRCh38", release="112")
    else:
        ann_registry = resources.get_registry()

    gtf_path = str(ann_registry.get_gtf_path())
    gene_annotations = extract_gene_annotations(gtf_path, verbosity=0)

    # ── Resolve evaluation gene set ────────────────────────────────
    gene_chroms = gene_chromosomes_from_dataframe(gene_annotations)

    if args.genes:
        test_genes = args.genes
        test_chroms = sorted(set(
            gene_chroms.get(g, "unknown") for g in test_genes if g in gene_chroms
        ))
        print(f"  Eval genes: {len(test_genes)} (user-specified)")
    elif args.test_chroms:
        test_chroms = [c if c.startswith("chr") else f"chr{c}" for c in args.test_chroms]
        test_genes = sorted(g for g, c in gene_chroms.items() if c in test_chroms)
        print(f"  Eval genes: {len(test_genes)} on {test_chroms}")
    else:
        gene_split = build_gene_split(gene_chroms, preset="spliceai", val_fraction=0.0)
        test_genes = sorted(gene_split.test_genes)
        test_chroms = sorted(set(
            gene_chroms.get(g, "unknown") for g in test_genes if g in gene_chroms
        ))
        print(f"  Eval genes: {len(test_genes)} (SpliceAI test: {test_chroms})")

    if args.max_genes:
        test_genes = test_genes[:args.max_genes]
        print(f"  Limited to {len(test_genes)} genes")

    # ── Resolve cache directory ─────────────────────────────────────
    cache_dir = args.cache_dir or args.checkpoint.parent / "gene_cache" / "test"

    # ── Build gene cache if requested ────────────────────────────────
    if args.build_cache:
        import pandas as pd
        from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
            build_gene_cache,
        )
        from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
            DenseFeatureExtractor, DenseFeatureConfig,
        )
        from agentic_spliceai.splice_engine.eval.streaming_metrics import (
            preflight_check,
        )

        fasta_path = str(resources.get_fasta_path())
        splice_sites_path = Path(ann_registry.stash) / "splice_sites_enhanced.tsv"
        if args.base_scores_dir:
            base_scores_dir = args.base_scores_dir
        else:
            base_scores_dir = resources.get_registry().get_base_model_eval_dir(
                "openspliceai"
            ) / "precomputed"

        # Fail fast if dependencies or data are missing
        preflight_check(
            needs_bigwig=True,
            needs_pyfaidx=True,
            fasta_path=fasta_path,
            base_scores_dir=base_scores_dir,
        )

        print(f"\n  Building test gene cache ({len(test_genes)} genes)...")
        print(f"    Cache dir:    {cache_dir}")
        print(f"    FASTA:        {fasta_path}")
        print(f"    Splice sites: {splice_sites_path}")
        print(f"    Base scores:  {base_scores_dir}")

        splice_sites_df = pd.read_csv(splice_sites_path, sep="\t")
        feat_config = DenseFeatureConfig(
            build="GRCh38",
            bigwig_cache_dir=args.bigwig_cache,
        )
        extractor = DenseFeatureExtractor(feat_config)

        t_cache = time.time()
        build_gene_cache(
            test_genes, splice_sites_df, fasta_path,
            base_scores_dir, extractor, gene_annotations,
            cache_dir=cache_dir,
        )
        extractor.close()
        print(f"    Cache built in {time.time() - t_cache:.1f}s\n")

    # ── Streaming evaluation ─────────────────────────────────────────
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz,
    )
    from agentic_spliceai.splice_engine.eval.streaming_metrics import (
        StreamingEvaluator, print_comparison_report,
    )
    from agentic_spliceai.splice_engine.eval.sequence_inference import (
        apply_temperature_blend,
    )

    min_length = 5001 + 400  # window_size + context_padding
    evaluator = StreamingEvaluator()

    # ── Ablation: resolve channel indices to zero ────────────────────
    zero_channel_indices = None
    if args.zero_channels:
        from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
            CHANNEL_NAMES,
        )
        if "all" in args.zero_channels:
            zero_channel_indices = list(range(len(CHANNEL_NAMES)))
            ablation_label = "all channels zeroed (sequence + base scores only)"
        else:
            zero_channel_indices = []
            for ch in args.zero_channels:
                if ch not in CHANNEL_NAMES:
                    print(f"ERROR: Unknown channel '{ch}'. Valid: {CHANNEL_NAMES}")
                    return 1
                zero_channel_indices.append(CHANNEL_NAMES.index(ch))
            ablation_label = f"zeroed: {', '.join(args.zero_channels)}"
        print(f"  Ablation: {ablation_label}")

    # ── Temperature scaling: resolve blend_alpha from model ─────────
    import torch
    blend_alpha = 0.5  # default (sigmoid(0))
    if hasattr(model, "blend_alpha"):
        blend_alpha = float(torch.sigmoid(model.blend_alpha).item())
        log.info("Model blend_alpha: %.4f (raw=%.4f)", blend_alpha, model.blend_alpha.item())

    # Temperature can be scalar (--temperature) or array (--calibrate-temperature)
    temperature = args.temperature or 1.0
    use_temperature = args.temperature is not None or args.calibrate_temperature

    # ── Temperature calibration on validation set ───────────────────
    if args.calibrate_temperature:
        from agentic_spliceai.splice_engine.eval.streaming_metrics import (
            TemperatureScaler,
        )

        val_cache_dir = args.val_cache_dir or args.checkpoint.parent / "gene_cache" / "val"
        if not val_cache_dir.exists():
            print(f"ERROR: Validation cache not found: {val_cache_dir}")
            print("  Build it first with --build-cache --test-chroms chr2 chr4 chr6 chr8 chr10")
            return 1

        # Use all genes available in the val cache directory.
        # The caller is responsible for building the val cache from the
        # right chromosomes (e.g. --test-chroms chr2 chr4 --cache-dir .../val).
        val_npz_files = sorted(val_cache_dir.glob("*.npz"))
        val_genes = [f.stem for f in val_npz_files]
        if not val_genes:
            print(f"ERROR: No .npz files found in {val_cache_dir}")
            return 1

        print(f"\n{'='*70}")
        print(f"Temperature Calibration ({len(val_genes)} validation genes)")
        print(f"{'='*70}\n")

        scaler = TemperatureScaler(subsample_rate=0.1)
        n_cal = 0
        for gene_id in val_genes:
            npz_path = val_cache_dir / f"{gene_id}.npz"
            if not npz_path.exists():
                continue
            data = _load_gene_npz(npz_path)
            if len(data["sequence"]) < min_length:
                del data
                continue

            logits = infer_full_gene(model, data, device=device, return_logits=True)
            scaler.collect(logits, data["base_scores"], data["labels"])
            del data, logits
            n_cal += 1

        if n_cal < 10:
            print(f"WARNING: Only {n_cal} validation genes found. "
                  "Results may be unreliable. Skipping calibration.")
        else:
            cal_result = scaler.fit(blend_alpha=blend_alpha)
            temperature = cal_result["temperature"]  # np.ndarray [3]
            T = temperature
            print(f"  Class-wise temperature: [donor={T[0]:.4f}, acceptor={T[1]:.4f}, neither={T[2]:.4f}]")
            print(f"  NLL: {cal_result['nll_before']:.4f} → {cal_result['nll_after']:.4f}")
            print(f"  ECE: {cal_result['ece_before']:.4f} → {cal_result['ece_after']:.4f}")
            print(f"  Validation genes: {n_cal}")
            print(f"  Positions used: {cal_result['n_positions']:,}")
            use_temperature = True

        del scaler

    _is_default_T = (
        np.isscalar(temperature) and temperature == 1.0
    ) if np.isscalar(temperature) else np.allclose(temperature, 1.0)

    if use_temperature and not _is_default_T:
        if np.isscalar(temperature):
            print(f"\n  Temperature scaling: T={temperature:.4f}, blend_alpha={blend_alpha:.4f}")
        else:
            T = temperature
            print(f"\n  Class-wise temperature: [donor={T[0]:.4f}, acceptor={T[1]:.4f}, neither={T[2]:.4f}]")
            print(f"  blend_alpha={blend_alpha:.4f}")

    print(f"\n{'='*70}")
    print("Evaluating: Meta model vs Base model (streaming)")
    print(f"{'='*70}\n")

    t0 = time.time()
    for i, gene_id in enumerate(test_genes):
        npz_path = cache_dir / f"{gene_id}.npz"
        if not npz_path.exists():
            evaluator.n_skipped += 1
            continue

        data = _load_gene_npz(npz_path)
        if len(data["sequence"]) < min_length:
            evaluator.n_skipped += 1
            del data
            continue

        # Ablation: zero out specified channels before inference
        if zero_channel_indices is not None:
            data["mm_features"][:, zero_channel_indices] = 0.0

        # Inference: one gene at a time
        if use_temperature and not _is_default_T:
            # Temperature scaling: get logits, then apply T + blend externally
            logits = infer_full_gene(model, data, device=device, return_logits=True)
            meta_probs = apply_temperature_blend(
                logits, data["base_scores"], temperature, blend_alpha,
            )
            del logits
        else:
            meta_probs = infer_full_gene(model, data, device=device)

        base_probs = data["base_scores"]
        labels = data["labels"]

        evaluator.update(meta_probs, base_probs, labels, gene_id)

        # Free immediately — only one gene's arrays in memory
        del data, meta_probs, base_probs, labels

        if (i + 1) % 100 == 0:
            mem_mb = evaluator.memory_usage_mb()
            print(f"  Processed {i+1}/{len(test_genes)} genes "
                  f"(accum: {mem_mb:.1f} MB)...")

    elapsed = time.time() - t0
    print(f"  Inference complete: {evaluator.n_genes} genes in {elapsed:.1f}s "
          f"({evaluator.n_skipped} skipped)")
    print(f"  Accumulator memory: {evaluator.memory_usage_mb():.1f} MB")

    if evaluator.n_genes == 0:
        print("ERROR: No genes evaluated. Check --cache-dir path.")
        return 1

    # ── Compute and display metrics ──────────────────────────────────
    results = evaluator.compute()
    results["model"] = cfg.variant
    results["checkpoint"] = str(args.checkpoint)
    results["annotation_source"] = ann_src
    results["test_chromosomes"] = test_chroms if 'test_chroms' in dir() else []
    if args.zero_channels:
        results["ablation"] = args.zero_channels
    if use_temperature and not _is_default_T:
        results["temperature"] = (
            temperature.tolist() if hasattr(temperature, "tolist") else temperature
        )
        results["blend_alpha"] = blend_alpha
        if args.calibrate_temperature and 'cal_result' in dir():
            cal_for_json = {**cal_result}
            if hasattr(cal_for_json["temperature"], "tolist"):
                cal_for_json["temperature"] = cal_for_json["temperature"].tolist()
            results["calibration"] = cal_for_json

    print_comparison_report(results)

    # ── Threshold sweep ──────────────────────────────────────────────
    if args.sweep_thresholds:
        from agentic_spliceai.splice_engine.eval.streaming_metrics import (
            print_threshold_analysis,
        )
        sweep = evaluator.sweep_thresholds()
        print_threshold_analysis(sweep)
        results["threshold_sweep"] = sweep

    # ── Save results ─────────────────────────────────────────────────
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use descriptive filename based on evaluation mode
    if args.zero_channels:
        suffix = "_".join(args.zero_channels).replace(",", "_")
        results_path = output_dir / f"eval_ablation_{suffix}.json"
    elif use_temperature and not _is_default_T:
        if np.isscalar(temperature):
            results_path = output_dir / f"eval_results_T{temperature:.2f}.json"
        else:
            results_path = output_dir / "eval_results_calibrated.json"
    else:
        results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    meta_m = results["meta_model"]
    base_m = results["base_model"]
    print(f"\n{'='*70}")
    print(f"Summary: Meta model {'improves' if meta_m['macro_pr_auc'] > base_m['macro_pr_auc'] else 'does not improve'} over base model")
    print(f"  Base PR-AUC: {base_m['macro_pr_auc']:.4f}")
    print(f"  Meta PR-AUC: {meta_m['macro_pr_auc']:.4f}")
    print(f"  FN reduction: {results['fn_reduction_pct']:+.1f}%")
    print(f"  FP reduction: {results['fp_reduction_pct']:+.1f}%")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

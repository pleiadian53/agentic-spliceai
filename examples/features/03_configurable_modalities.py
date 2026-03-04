#!/usr/bin/env python
"""Feature Engineering Example 3: Configurable Modalities.

Demonstrates how to tune modality parameters using typed config objects:
1. Default config (context_window=2) vs wider context (context_window=5)
2. Enable/disable feature groups (gradients, comparative)
3. Side-by-side comparison of column counts and feature values

Each modality has a typed config dataclass with discoverable parameters.
Users only override what they need — sensible defaults are always provided.

Usage:
    python 03_configurable_modalities.py --gene BRCA1
    python 03_configurable_modalities.py --gene TP53 --context-window 10

Example:
    python 03_configurable_modalities.py --gene BRCA1
"""

import argparse
import sys
import time

import polars as pl

from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner
from agentic_spliceai.splice_engine.features import FeaturePipeline, FeaturePipelineConfig
from agentic_spliceai.splice_engine.features.modalities.base_scores import BaseScoreConfig


def main():
    """Run configurable modalities example."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Example 3: Configurable Modalities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gene", default="BRCA1",
        help="Gene symbol (default: BRCA1)",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)",
    )
    parser.add_argument(
        "--context-window", type=int, default=5,
        help="Wider context window for comparison (default: 5)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Feature Engineering Example 3: Configurable Modalities")
    print("=" * 70)
    print(f"\n🧬 Gene: {args.gene}")
    print(f"🔧 Model: {args.model}")
    print(f"📏 Comparison: context_window=2 (default) vs {args.context_window}")

    # ---------------------------------------------------------------
    # Step 1: Run base-layer prediction
    # ---------------------------------------------------------------
    print(f"\n📡 Running base-layer prediction for {args.gene}...")
    runner = BaseModelRunner()
    result = runner.run_single_model(
        model_name=args.model,
        target_genes=[args.gene],
        test_name=f"config_example_{args.gene}",
        mode="test",
        coverage="gene_subset",
        verbosity=0,
    )

    if not result.success:
        print(f"❌ Prediction failed: {result.error}")
        return 1

    predictions = result.positions
    print(f"   {predictions.height:,} positions")

    # ---------------------------------------------------------------
    # Step 2: Default config (context_window=2)
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Config A: Default (context_window=2)")
    print("-" * 70)

    config_a = FeaturePipelineConfig(
        modalities=["base_scores"],
        verbosity=0,
    )
    pipeline_a = FeaturePipeline(config_a)
    t0 = time.time()
    result_a = pipeline_a.transform(predictions)
    dt_a = time.time() - t0

    ctx_cols_a = [c for c in result_a.columns if c.startswith("context_score_")]
    print(f"   Total columns: {result_a.width}")
    print(f"   Context score columns ({len(ctx_cols_a)}): {ctx_cols_a}")
    print(f"   Runtime: {dt_a:.2f}s")

    # ---------------------------------------------------------------
    # Step 3: Wider context window
    # ---------------------------------------------------------------
    print(f"\n" + "-" * 70)
    print(f"Config B: Wider (context_window={args.context_window})")
    print("-" * 70)

    config_b = FeaturePipelineConfig(
        modalities=["base_scores"],
        modality_configs={
            "base_scores": BaseScoreConfig(context_window=args.context_window),
        },
        verbosity=0,
    )
    pipeline_b = FeaturePipeline(config_b)
    t0 = time.time()
    result_b = pipeline_b.transform(predictions)
    dt_b = time.time() - t0

    ctx_cols_b = [c for c in result_b.columns if c.startswith("context_score_")]
    print(f"   Total columns: {result_b.width}")
    print(f"   Context score columns ({len(ctx_cols_b)}): {ctx_cols_b}")
    print(f"   Runtime: {dt_b:.2f}s")

    extra_cols = set(result_b.columns) - set(result_a.columns)
    print(f"   Extra columns vs default: {sorted(extra_cols)}")

    # ---------------------------------------------------------------
    # Step 4: Compare a strong splice site across configs
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Side-by-Side: Top Donor Position")
    print("-" * 70)

    top_pos = result_a.sort("donor_prob", descending=True).head(1)["position"][0]
    row_a = result_a.filter(pl.col("position") == top_pos).row(0, named=True)
    row_b = result_b.filter(pl.col("position") == top_pos).row(0, named=True)

    print(f"   Position: {top_pos}")
    print(f"   donor_prob: {row_a['donor_prob']:.6f}")
    print(f"\n   {'Feature':<25s} {'Default (w=2)':>15s} {'Wider (w={})':>15s}".format(
        args.context_window
    ))
    print(f"   {'─' * 25} {'─' * 15} {'─' * 15}")

    compare_cols = ["donor_is_local_peak", "donor_surge_ratio",
                    "donor_signal_strength", "donor_peak_height_ratio"]
    for col in compare_cols:
        va = row_a.get(col, "N/A")
        vb = row_b.get(col, "N/A")
        if isinstance(va, float):
            print(f"   {col:<25s} {va:>15.4f} {vb:>15.4f}")
        else:
            print(f"   {col:<25s} {va!s:>15s} {vb!s:>15s}")

    # Context scores comparison
    print(f"\n   Context scores at this position:")
    for i in range(1, args.context_window + 1):
        cm = f"context_score_m{i}"
        cp = f"context_score_p{i}"
        va_m = row_a.get(cm, "—")
        vb_m = row_b.get(cm, "—")
        va_p = row_a.get(cp, "—")
        vb_p = row_b.get(cp, "—")
        mark_a_m = f"{va_m:.6f}" if isinstance(va_m, float) else "—"
        mark_b_m = f"{vb_m:.6f}" if isinstance(vb_m, float) else "—"
        mark_a_p = f"{va_p:.6f}" if isinstance(va_p, float) else "—"
        mark_b_p = f"{vb_p:.6f}" if isinstance(vb_p, float) else "—"
        print(f"     m{i}: {mark_a_m:>12s} → {mark_b_m:>12s}   "
              f"p{i}: {mark_a_p:>12s} → {mark_b_p:>12s}")

    # ---------------------------------------------------------------
    # Step 5: Minimal config (disable feature groups)
    # ---------------------------------------------------------------
    print(f"\n" + "-" * 70)
    print("Config C: Minimal (no gradients, no comparative)")
    print("-" * 70)

    config_c = FeaturePipelineConfig(
        modalities=["base_scores"],
        modality_configs={
            "base_scores": BaseScoreConfig(
                include_gradients=False,
                include_comparative=False,
            ),
        },
        verbosity=0,
    )
    pipeline_c = FeaturePipeline(config_c)
    result_c = pipeline_c.transform(predictions)

    print(f"   Total columns: {result_c.width} (vs {result_a.width} default)")
    removed = set(result_a.columns) - set(result_c.columns)
    print(f"   Removed {len(removed)} gradient/comparative columns")
    kept = [c for c in result_c.columns if c not in predictions.columns]
    print(f"   Kept features: {kept}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 Configuration Comparison:")
    print(f"   Config A (default w=2):      {result_a.width} columns")
    print(f"   Config B (wider w={args.context_window}):       {result_b.width} columns")
    print(f"   Config C (minimal):          {result_c.width} columns")
    print(f"\n💡 Next: Try 04_genome_scale_workflow.py for production-scale features")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

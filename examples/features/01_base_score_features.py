#!/usr/bin/env python
"""Feature Engineering Example 1: Base Score Features.

Demonstrates the FeaturePipeline with the base_scores modality only.
No FASTA, GTF, or annotation files needed — works directly on base
layer predictions for any gene.

Key concepts:
1. Run a base-layer prediction for a single gene
2. Apply FeaturePipeline with base_scores modality
3. Inspect derived features: context scores, gradients, probability features
4. Verify FeatureSchema compatibility (43/43 required columns)

Usage:
    python 01_base_score_features.py --gene BRCA1
    python 01_base_score_features.py --gene TP53 --model spliceai

Example:
    python 01_base_score_features.py --gene BRCA1
"""

import argparse
import sys
import time

import polars as pl

from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner
from agentic_spliceai.splice_engine.features import FeaturePipeline, FeaturePipelineConfig
from agentic_spliceai.splice_engine.meta_layer.core.feature_schema import DEFAULT_SCHEMA


def main():
    """Run base score feature engineering example."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Example 1: Base Score Features",
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
    args = parser.parse_args()

    print("=" * 70)
    print("Feature Engineering Example 1: Base Score Features")
    print("=" * 70)
    print(f"\n🧬 Gene: {args.gene}")
    print(f"🔧 Model: {args.model}")

    # ---------------------------------------------------------------
    # Step 1: Run base-layer prediction
    # ---------------------------------------------------------------
    print(f"\n📡 Running base-layer prediction for {args.gene}...")
    t0 = time.time()

    runner = BaseModelRunner()
    result = runner.run_single_model(
        model_name=args.model,
        target_genes=[args.gene],
        test_name=f"features_example_{args.gene}",
        mode="test",
        coverage="gene_subset",
        verbosity=0,
    )

    if not result.success:
        print(f"❌ Prediction failed: {result.error}")
        return 1

    predictions = result.positions
    print(f"   {predictions.height:,} positions predicted in {time.time() - t0:.1f}s")
    print(f"   Input columns ({predictions.width}): {predictions.columns}")

    # ---------------------------------------------------------------
    # Step 2: Apply FeaturePipeline with base_scores only
    # ---------------------------------------------------------------
    print("\n🔬 Applying FeaturePipeline (base_scores modality)...")
    t1 = time.time()

    config = FeaturePipelineConfig(
        modalities=["base_scores"],
        verbosity=0,
    )
    pipeline = FeaturePipeline(config)
    enriched = pipeline.transform(predictions)

    new_cols = [c for c in enriched.columns if c not in predictions.columns]
    print(f"   Added {len(new_cols)} feature columns in {time.time() - t1:.1f}s")
    print(f"   Total: {enriched.width} columns ({predictions.width} input + {len(new_cols)} derived)")

    # ---------------------------------------------------------------
    # Step 3: Display feature groups
    # ---------------------------------------------------------------
    schema = pipeline.get_output_schema()
    print("\n📊 Feature Groups:")
    for modality_name, cols in schema.items():
        print(f"   {modality_name}: {len(cols)} columns")

        # Group by prefix for readability
        groups: dict[str, list[str]] = {}
        for c in cols:
            prefix = c.rsplit("_", 1)[0] if "_" in c else c
            # Simplify grouping
            if c.startswith("context_score"):
                key = "context_score_*"
            elif c.startswith("donor_diff") or c.startswith("acceptor_diff"):
                key = "*_diff_*"
            elif c.endswith("_score"):
                key = "*_score (aliases)"
            else:
                key = c
            groups.setdefault(key, []).append(c)

        for group, members in groups.items():
            if len(members) > 1:
                print(f"     {group} ({len(members)} cols)")
            else:
                print(f"     {members[0]}")

    # ---------------------------------------------------------------
    # Step 4: Spotlight on strong splice sites
    # ---------------------------------------------------------------
    print("\n🎯 Top Splice Sites (spotlight):")

    # Top donor
    top_donor = enriched.sort("donor_prob", descending=True).head(1)
    if top_donor.height > 0:
        row = top_donor.row(0, named=True)
        print(f"\n   Top Donor: {row['gene_name']} position {row['position']}")
        print(f"     donor_prob:           {row['donor_prob']:.6f}")
        print(f"     donor_is_local_peak:  {row['donor_is_local_peak']}")
        print(f"     donor_surge_ratio:    {row['donor_surge_ratio']:.2f}")
        print(f"     donor_signal_strength: {row['donor_signal_strength']:.6f}")
        print(f"     context_score_m1:     {row['context_score_m1']:.6f}")
        print(f"     context_score_p1:     {row['context_score_p1']:.6f}")
        print(f"     probability_entropy:  {row['probability_entropy']:.6f}")

    # Top acceptor
    top_acc = enriched.sort("acceptor_prob", descending=True).head(1)
    if top_acc.height > 0:
        row = top_acc.row(0, named=True)
        print(f"\n   Top Acceptor: {row['gene_name']} position {row['position']}")
        print(f"     acceptor_prob:           {row['acceptor_prob']:.6f}")
        print(f"     acceptor_is_local_peak:  {row['acceptor_is_local_peak']}")
        print(f"     acceptor_surge_ratio:    {row['acceptor_surge_ratio']:.2f}")
        print(f"     acceptor_signal_strength: {row['acceptor_signal_strength']:.6f}")

    # ---------------------------------------------------------------
    # Step 5: FeatureSchema compatibility check
    # ---------------------------------------------------------------
    schema_cols = set(DEFAULT_SCHEMA.get_all_feature_cols())
    output_cols = set(enriched.columns)
    covered = schema_cols & output_cols
    missing = schema_cols - output_cols

    print(f"\n📋 FeatureSchema Compatibility:")
    print(f"   Schema expects: {len(schema_cols)} feature columns")
    print(f"   Pipeline covers: {len(covered)}/{len(schema_cols)}")
    if missing:
        print(f"   Missing: {sorted(missing)}")
    else:
        print(f"   ✅ All schema columns covered!")

    print("\n" + "=" * 70)
    print(f"💡 Next: Try 02_annotation_and_genomic.py for multi-modal features")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

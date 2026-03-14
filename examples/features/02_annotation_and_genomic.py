#!/usr/bin/env python
"""Feature Engineering Example 2: Multi-Modal with Annotations.

Demonstrates composing multiple modalities:
1. base_scores — derived probability and gradient features
2. annotation — ground truth splice_type labels from GTF
3. genomic — positional features (relative position, gene distances)
4. conservation (optional) — evolutionary constraint (PhyloP / PhastCons)

Shows how the pipeline auto-resolves resource paths (splice_sites_enhanced.tsv,
conservation bigWig tracks) from the genomic registry based on the base model.

Usage:
    python 02_annotation_and_genomic.py --gene TP53
    python 02_annotation_and_genomic.py --gene BRCA1 --model openspliceai
    python 02_annotation_and_genomic.py --gene TP53 --conservation

Example:
    python 02_annotation_and_genomic.py --gene TP53
"""

import argparse
import sys
import time

import polars as pl

from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner
from agentic_spliceai.splice_engine.features import FeaturePipeline, FeaturePipelineConfig


def main():
    """Run multi-modal feature engineering example."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Example 2: Multi-Modal with Annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gene", default="TP53",
        help="Gene symbol (default: TP53)",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)",
    )
    parser.add_argument(
        "--conservation", action="store_true",
        help="Include conservation modality (PhyloP / PhastCons). "
             "Requires pyBigWig; fetches scores from UCSC bigWig tracks.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Feature Engineering Example 2: Multi-Modal with Annotations")
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
    print(f"   {predictions.height:,} positions in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # Step 2: Apply pipeline with modalities
    # ---------------------------------------------------------------
    modalities = ["base_scores", "annotation", "genomic"]
    modality_configs = {}

    if args.conservation:
        from agentic_spliceai.splice_engine.features.modalities.conservation import (
            ConservationConfig,
        )
        modalities.append("conservation")
        modality_configs["conservation"] = ConservationConfig(
            base_model=args.model,
            window=10,
        )

    print(f"\n🔬 Applying FeaturePipeline ({' + '.join(modalities)})...")
    t1 = time.time()

    config = FeaturePipelineConfig(
        base_model=args.model,
        modalities=modalities,
        modality_configs=modality_configs,
        verbosity=0,
    )
    pipeline = FeaturePipeline(config)
    enriched = pipeline.transform(predictions)

    n_new = enriched.width - predictions.width
    print(f"   Added {n_new} columns in {time.time() - t1:.1f}s")
    print(f"   Total: {enriched.width} columns")

    # Show what each modality contributed
    schema = pipeline.get_output_schema()
    for name, cols in schema.items():
        print(f"   {name}: +{len(cols)} columns")

    # ---------------------------------------------------------------
    # Step 3: Splice site label distribution
    # ---------------------------------------------------------------
    print("\n📊 Splice Type Distribution:")
    label_counts = enriched.group_by("splice_type").len().sort("len", descending=True)
    for row in label_counts.iter_rows(named=True):
        label = row["splice_type"] if row["splice_type"] else "(neither)"
        pct = 100 * row["len"] / enriched.height
        print(f"   {label:12s}: {row['len']:>8,} positions ({pct:.2f}%)")

    # ---------------------------------------------------------------
    # Step 4: Annotated splice sites with their features
    # ---------------------------------------------------------------
    splice_sites = enriched.filter(pl.col("splice_type") != "")
    print(f"\n🎯 Known Splice Sites: {splice_sites.height} positions")

    if splice_sites.height > 0:
        # Show top donor sites
        donors = splice_sites.filter(pl.col("splice_type") == "donor").sort(
            "donor_prob", descending=True
        )
        print(f"\n   Top Donor Sites ({donors.height} total):")
        for row in donors.head(3).iter_rows(named=True):
            print(
                f"     pos={row['position']:,}  "
                f"donor_prob={row['donor_prob']:.4f}  "
                f"peak={row['donor_is_local_peak']}  "
                f"rel_pos={row['relative_gene_position']:.3f}"
            )

        # Show top acceptor sites
        acceptors = splice_sites.filter(pl.col("splice_type") == "acceptor").sort(
            "acceptor_prob", descending=True
        )
        print(f"\n   Top Acceptor Sites ({acceptors.height} total):")
        for row in acceptors.head(3).iter_rows(named=True):
            print(
                f"     pos={row['position']:,}  "
                f"acceptor_prob={row['acceptor_prob']:.4f}  "
                f"peak={row['acceptor_is_local_peak']}  "
                f"rel_pos={row['relative_gene_position']:.3f}"
            )

    # ---------------------------------------------------------------
    # Step 5: Genomic feature summary
    # ---------------------------------------------------------------
    print("\n📈 Genomic Feature Summary:")
    for col in ["relative_gene_position", "distance_to_gene_start", "distance_to_gene_end"]:
        if col in enriched.columns:
            stats = enriched[col].describe()
            mean_val = enriched[col].mean()
            min_val = enriched[col].min()
            max_val = enriched[col].max()
            print(f"   {col}:")
            print(f"     min={min_val:.1f}  mean={mean_val:.1f}  max={max_val:.1f}")

    # ---------------------------------------------------------------
    # Step 6: Conservation summary (if enabled)
    # ---------------------------------------------------------------
    if args.conservation and "phylop_score" in enriched.columns:
        print("\n🧬 Conservation Summary (PhyloP / PhastCons):")
        splice = enriched.filter(pl.col("splice_type") != "")
        bg = enriched.filter(pl.col("splice_type") == "").sample(
            n=min(200, enriched.filter(pl.col("splice_type") == "").height),
            seed=42,
        )
        for label, subset in [("Splice sites", splice), ("Background", bg)]:
            if subset.height > 0:
                phylop_mean = subset["phylop_score"].mean()
                phastcons_mean = subset["phastcons_score"].mean()
                print(f"   {label:15s} (n={subset.height:>5,}): "
                      f"PhyloP={phylop_mean:>6.2f}  PhastCons={phastcons_mean:.3f}")

    print("\n" + "=" * 70)
    print(f"💡 Next: Try 03_configurable_modalities.py to tune parameters")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

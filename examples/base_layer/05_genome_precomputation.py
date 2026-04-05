#!/usr/bin/env python
"""Genome-scale base model precomputation for meta-layer training.

Runs a base model (OpenSpliceAI, SpliceAI, or foundation model-derived
classifiers) on gene annotations and saves per-nucleotide splice site
scores as prediction parquets.  These artifacts serve as input for the
meta-layer (both position-level M*-P and sequence-level M*-S models).

Supports multiple annotation sources on the same genome build:
- **MANE** (default for OpenSpliceAI): canonical protein-coding transcripts
- **Ensembl**: comprehensive annotations including alternative isoforms
  (enables M2a evaluation at Ensembl-only splice sites)

Output directory is resolved automatically based on the annotation source
and base model:
    data/{annotation_source}/{build}/{model}_eval/precomputed/

Usage:
    # OpenSpliceAI on MANE (default)
    python 05_genome_precomputation.py --all --chunk-size 500

    # OpenSpliceAI on Ensembl (for M2a evaluation)
    python 05_genome_precomputation.py --all --chunk-size 500 \\
        --annotation-source ensembl

    # SpliceAI on Ensembl/GRCh37
    python 05_genome_precomputation.py --all --model spliceai

    # Specific chromosomes, resume after interruption
    python 05_genome_precomputation.py --chromosomes chr21 chr22 --resume

    # Custom GTF (overrides annotation source resolution)
    python 05_genome_precomputation.py --chromosomes chr22 \\
        --gtf /path/to/custom.gtf
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.models.config import (
    create_workflow_config,
)
from agentic_spliceai.splice_engine.base_layer.workflows import PredictionWorkflow


ALL_CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def main():
    parser = argparse.ArgumentParser(
        description="Genome-scale base model precomputation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Chromosomes to precompute (e.g., chr22, or 21 22)",
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_chromosomes",
        help="Precompute all canonical chromosomes (chr1-22, chrX, chrY)",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model (default: openspliceai)",
    )
    parser.add_argument(
        "--annotation-source", default=None,
        choices=["mane", "ensembl"],
        help="Gene annotation source.  Default: model's training source "
             "(mane for openspliceai, ensembl for spliceai).  Override to "
             "run a model on a different annotation set.",
    )
    parser.add_argument(
        "--gtf", type=Path, default=None,
        help="Explicit GTF path (overrides --annotation-source resolution)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Genes per chunk.  Default: 500 for MANE (~19K genes), "
             "100 for Ensembl (~57K genes with larger gene spans).  "
             "Smaller chunks use less memory per iteration.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoints",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: auto-resolved from registry)",
    )
    args = parser.parse_args()

    if not args.chromosomes and not args.all_chromosomes:
        parser.error("at least one of --chromosomes or --all is required")

    # Resolve chromosome list
    if args.all_chromosomes:
        chromosomes = ALL_CANONICAL
    else:
        chromosomes = [
            c if c.startswith("chr") else f"chr{c}" for c in args.chromosomes
        ]

    # Auto-select chunk size based on annotation source.
    # Ensembl genes are larger on average (alternative transcripts extend
    # gene boundaries), so per-chunk memory is higher.
    chunk_size = args.chunk_size
    if chunk_size is None:
        effective_source = (args.annotation_source or "mane").lower()
        chunk_size = 100 if effective_source == "ensembl" else 500
        print(f"  Auto chunk size: {chunk_size} (for {effective_source})")

    # Build config
    config_kwargs = dict(
        base_model=args.model,
        chunk_size=chunk_size,
        mode="production",
        resume=args.resume,
    )
    if args.annotation_source:
        config_kwargs["override_annotation_source"] = args.annotation_source
    if args.gtf:
        config_kwargs["gtf_file"] = str(args.gtf)
    if args.output_dir:
        config_kwargs["output_dir"] = str(args.output_dir)

    config = create_workflow_config(**config_kwargs)

    # Determine effective annotation source for display
    effective_source = args.annotation_source or config.annotation_source

    # Warn about cross-build combinations.  All base models are
    # sequence-based so cross-build predictions are valid (the model
    # sees DNA, not coordinates), but the user should be aware.
    MODEL_NATIVE_BUILD = {
        "openspliceai": ("GRCh38", "mane"),
        "spliceai": ("GRCh37", "ensembl"),
    }
    native_build, native_source = MODEL_NATIVE_BUILD.get(
        args.model, (config.genomic_build, effective_source)
    )
    if effective_source != native_source:
        print(f"NOTE: {args.model} was trained on {native_source}/{native_build}, "
              f"but running on {effective_source}/{config.genomic_build} annotations.")
        if config.genomic_build != native_build:
            print(f"  Build mismatch: model is {native_build}-native, "
                  f"sequences are {config.genomic_build}.")
            print(f"  This is valid (sequence-based model) but may show "
                  f"build-specific performance differences.")
        print()

    print("=" * 70)
    print("Genome Precomputation for Meta-Layer Training")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Annotations: {effective_source} / {config.genomic_build}")
    if args.gtf:
        print(f"  GTF:         {args.gtf}")
    print(f"  Chromosomes: {len(chromosomes)}")
    print(f"  Chunk size:  {args.chunk_size} genes")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {config.output_dir}")

    # Run workflow — per-chromosome with incremental parquet saves.
    #
    # Each chromosome's predictions are saved as a separate parquet file
    # (e.g., predictions_chr1.parquet) immediately after processing.
    # This provides:
    # 1. Memory bounding — only one chromosome in RAM at a time
    # 2. Crash recovery — completed chromosomes are preserved on disk
    # 3. Resume support — existing parquets are skipped on restart
    import gc
    import time as _time
    import polars as pl
    from agentic_spliceai.splice_engine.utils.memory_monitor import get_rss_bytes_precise

    def _log_mem(label: str) -> None:
        rss_gb = get_rss_bytes_precise() / (1024 ** 3)
        print(f"  [MEM] {label}: {rss_gb:.1f} GB RSS", flush=True)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = _time.time()
    total_positions = 0
    total_processed = 0
    total_failed = 0
    skipped_chroms = 0

    _log_mem("start")

    for chrom in chromosomes:
        # Resume: skip chromosomes with existing parquet
        parquet_path = output_dir / f"predictions_{chrom}.parquet"
        if args.resume and parquet_path.exists():
            try:
                existing = pl.scan_parquet(parquet_path).select(
                    pl.len()
                ).collect().item()
                print(f"  {chrom}: skipped (resumed, {existing:,} positions)")
                total_positions += existing
                skipped_chroms += 1
                continue
            except Exception:
                pass  # Corrupted parquet, reprocess

        chrom_start = _time.time()
        print(f"\n{'─'*70}")
        print(f"  Chromosome: {chrom}")
        print(f"{'─'*70}")
        _log_mem(f"{chrom} before workflow")

        workflow = PredictionWorkflow(config)
        result = workflow.run(chromosomes=[chrom])
        _log_mem(f"{chrom} after workflow")

        if not result.success:
            print(f"  {chrom}: FAILED — {result.error}")
            total_failed += 1
            del workflow, result
            gc.collect()
            continue

        summary = result.manifest.get_summary()
        elapsed = _time.time() - chrom_start

        # In streaming mode, predictions are saved as chunk TSV files
        # but not accumulated in result.predictions.  Aggregate from
        # chunk files into per-chromosome parquet.
        if result.predictions.height > 0:
            result.predictions.write_parquet(parquet_path)
            n_pos = result.predictions.height
        else:
            # Streaming mode: concat chunk files from disk
            chunk_dir = Path(config.output_dir)
            chunk_files = sorted(chunk_dir.glob("predictions_chunk_*.tsv"))
            if chunk_files:
                chunk_dfs = [pl.read_csv(f, separator="\t") for f in chunk_files]
                combined = pl.concat(chunk_dfs)
                combined.write_parquet(parquet_path)
                n_pos = combined.height
                del chunk_dfs, combined
                # Clean up chunk TSVs now that parquet is written
                for f in chunk_files:
                    f.unlink()
            else:
                n_pos = 0

        if n_pos > 0:
            print(f"  {chrom}: {n_pos:,} positions, "
                  f"{summary['processed_genes']} genes ({elapsed:.0f}s)")
            print(f"    Saved: {parquet_path}")
            total_positions += n_pos
            total_processed += summary["processed_genes"]
            total_failed += summary["failed_genes"]
        else:
            print(f"  {chrom}: 0 positions (no genes?)")

        # Aggressively free memory before next chromosome
        del workflow, result
        gc.collect()
        _log_mem(f"{chrom} after gc")

    total_elapsed = _time.time() - total_start

    print(f"\n{'='*70}")
    print("Precomputation complete!")
    print(f"  Total positions: {total_positions:,}")
    print(f"  Processed:       {total_processed} genes")
    print(f"  Failed:          {total_failed} genes")
    print(f"  Skipped (resume):{skipped_chroms} chromosomes")
    print(f"  Runtime:         {total_elapsed:.0f}s")
    print(f"  Output:          {output_dir}")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

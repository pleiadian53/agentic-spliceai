#!/usr/bin/env python
"""Feature Engineering Example 5: Hands-On Multimodal Exploration.

Explore three signal modalities that complement base splice-site predictions:

1. **Conservation** (PhyloP / PhastCons) — Evolutionary constraint at splice sites
   vs random intronic positions. Conserved splice sites are under strong purifying
   selection; novel/alternative sites show weaker conservation.

2. **Epigenetic marks** (H3K36me3, H3K4me3) — Histone modifications that mark
   exon bodies (H3K36me3) and promoter regions (H3K4me3). These signals help
   distinguish actively used exons from genomic background.

3. **RNA-seq junction evidence** — Splice junction read counts from STAR aligner
   output. Direct experimental evidence of splicing activity at each site.

All bigWig data is accessed remotely via UCSC/ENCODE URLs — no local downloads
required (pyBigWig streams on demand). TP53 (chr17) is used as the example gene.

Prerequisites:
    pip install pyBigWig   # must be compiled against your NumPy version

Usage:
    python 05_multimodal_exploration.py
    python 05_multimodal_exploration.py --gene BRCA1
    python 05_multimodal_exploration.py --gene TP53 --n-random 50

Note: First run may be slow (~10-30s) as pyBigWig fetches remote bigWig headers.
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Remote bigWig URLs (no local files needed) ──────────────────────────
# These are publicly hosted by UCSC and ENCODE. pyBigWig opens them via HTTP
# range requests — only the regions you query are downloaded.

BIGWIG_URLS = {
    # Conservation: 100-way vertebrate alignment scores
    #   PhyloP: per-base conservation score. Positive = conserved (purifying
    #   selection), negative = fast-evolving (positive selection). Range ~[-20, 10].
    "phylop100": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
    #   PhastCons: probability that each base is in a conserved element.
    #   Range [0, 1]. Smoother than PhyloP — captures conserved *blocks*.
    "phastcons100": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
    # Epigenetic marks (ENCODE, K562 cell line, GRCh38, fold-change over control)
    # NOTE: Use direct S3 URLs, not @@download (ENCODE redirects break pyBigWig).
    # To find these: GET https://www.encodeproject.org/files/{ACC}/?format=json
    # → cloud_metadata.url
    #
    #   H3K36me3: Marks actively transcribed exon bodies. Deposited by SETD2
    #   during Pol II elongation. Strong signal = exon is being transcribed.
    "h3k36me3": (
        "https://encode-public.s3.amazonaws.com/2022/06/14/"
        "271813fb-7a9f-4f0c-8ec0-de92c16248d3/ENCFF975VUL.bigWig"
    ),
    #   H3K4me3: Marks active promoters / transcription start sites.
    #   Useful for identifying gene activity, not splice sites directly.
    "h3k4me3": (
        "https://encode-public.s3.amazonaws.com/2020/10/20/"
        "9bec9b64-3357-461b-a33e-08f3c72898a0/ENCFF941IYM.bigWig"
    ),
}


def check_pybigwig() -> bool:
    """Verify pyBigWig is importable and functional."""
    try:
        import pyBigWig  # noqa: F401
        return True
    except ImportError:
        log.error(
            "pyBigWig not installed. Install with:\n"
            "  pip install --no-cache-dir pyBigWig\n"
            "If you get ABI errors, force-reinstall against your NumPy version:\n"
            "  pip install --no-cache-dir --force-reinstall pyBigWig"
        )
        return False


def get_splice_sites(gene: str, model: str = "openspliceai") -> pl.DataFrame:
    """Load annotated splice sites for a gene using the core data prep infrastructure.

    Returns a DataFrame with columns: chrom, position, strand, splice_type,
    gene_name, transcript_id, exon_number.
    """
    from agentic_spliceai.splice_engine.base_layer.data import (
        prepare_splice_site_annotations,
    )
    from agentic_spliceai.splice_engine.resources import get_model_resources

    resources = get_model_resources(model)
    output_dir = resources.get_annotations_dir()

    result = prepare_splice_site_annotations(
        output_dir=output_dir,
        genes=[gene],
        build=resources.build,
        annotation_source=resources.annotation_source,
        verbosity=0,
    )
    if not result["success"]:
        raise RuntimeError(f"Failed to load splice sites for {gene}")

    df = result["splice_sites_df"]
    return df.filter(pl.col("gene_name") == gene)


def sample_intronic_positions(
    chrom: str,
    gene_start: int,
    gene_end: int,
    splice_positions: set[int],
    n: int = 30,
    margin: int = 50,
    seed: int = 42,
) -> list[int]:
    """Sample random positions within the gene body that are NOT near splice sites.

    Positions are at least `margin` bp away from any known splice site — these
    serve as a baseline for comparing conservation / epigenetic signals.
    """
    rng = random.Random(seed)
    candidates = []
    for _ in range(n * 20):  # oversample, then filter
        pos = rng.randint(gene_start + margin, gene_end - margin)
        if all(abs(pos - sp) >= margin for sp in splice_positions):
            candidates.append(pos)
        if len(candidates) >= n:
            break
    return candidates[:n]


# ═══════════════════════════════════════════════════════════════════════════
# Modality 1: Conservation Scores
# ═══════════════════════════════════════════════════════════════════════════

def explore_conservation(
    chrom: str,
    splice_positions: list[int],
    random_positions: list[int],
    window: int = 10,
) -> dict:
    """Compare conservation scores at splice sites vs random intronic positions.

    Biology: Splice sites (especially the GT/AG dinucleotides at donor/acceptor
    boundaries) are among the most conserved elements in the genome. The
    extended splice site consensus (~6bp at donor, ~20bp at acceptor) also
    shows strong conservation. Random intronic positions should show near-zero
    PhyloP scores on average.

    Args:
        chrom: Chromosome name (e.g., 'chr17')
        splice_positions: Known splice site positions
        random_positions: Random intronic positions (control)
        window: Half-window size for averaging (total = 2*window+1 bp)

    Returns:
        Dict with 'splice' and 'random' summary statistics for each track.
    """
    import pyBigWig

    results = {}

    for track_name, url in [
        ("phylop100", BIGWIG_URLS["phylop100"]),
        ("phastcons100", BIGWIG_URLS["phastcons100"]),
    ]:
        log.info(f"  Opening {track_name} (remote)...")
        bw = pyBigWig.open(url)

        def _get_scores(positions: list[int]) -> np.ndarray:
            scores = []
            for pos in positions:
                start = max(0, pos - window)
                end = pos + window + 1
                vals = bw.values(chrom, start, end)
                if vals is not None:
                    # Replace NaN with 0 (uncovered regions)
                    arr = np.array(vals, dtype=np.float64)
                    arr = np.nan_to_num(arr, nan=0.0)
                    scores.append(np.mean(arr))
                else:
                    scores.append(0.0)
            return np.array(scores)

        splice_scores = _get_scores(splice_positions)
        random_scores = _get_scores(random_positions)

        bw.close()

        results[track_name] = {
            "splice_mean": float(np.mean(splice_scores)),
            "splice_std": float(np.std(splice_scores)),
            "splice_median": float(np.median(splice_scores)),
            "random_mean": float(np.mean(random_scores)),
            "random_std": float(np.std(random_scores)),
            "random_median": float(np.median(random_scores)),
            "splice_scores": splice_scores,
            "random_scores": random_scores,
        }

    return results


def print_conservation_results(results: dict) -> None:
    """Pretty-print conservation comparison."""
    for track in ["phylop100", "phastcons100"]:
        r = results[track]
        label = "PhyloP (100-way)" if "phylop" in track else "PhastCons (100-way)"
        unit = "score" if "phylop" in track else "probability"

        print(f"\n  {label} ({unit}):")
        print(f"    {'':20s} {'Mean':>10s} {'Median':>10s} {'Std':>10s}")
        print(f"    {'Splice sites':<20s} {r['splice_mean']:>10.3f} "
              f"{r['splice_median']:>10.3f} {r['splice_std']:>10.3f}")
        print(f"    {'Random intronic':<20s} {r['random_mean']:>10.3f} "
              f"{r['random_median']:>10.3f} {r['random_std']:>10.3f}")
        ratio = r["splice_mean"] / max(abs(r["random_mean"]), 1e-6)
        print(f"    Splice/Random ratio: {ratio:.1f}x")


# ═══════════════════════════════════════════════════════════════════════════
# Modality 2: Epigenetic Marks (Histone Modifications)
# ═══════════════════════════════════════════════════════════════════════════

def explore_epigenetic(
    chrom: str,
    splice_sites_df: pl.DataFrame,
    gene_start: int,
    gene_end: int,
    window: int = 500,
) -> dict:
    """Compare histone mark signal at exon bodies vs intron bodies.

    Biology:
    - H3K36me3 is deposited on exon bodies during transcription by SETD2,
      recruited by the spliceosome. Exons that are actively spliced show
      HIGHER H3K36me3 signal than surrounding introns. This is one of the
      clearest epigenetic marks of exon usage.
    - H3K4me3 marks active promoters (TSS regions). It should be enriched
      at the 5' end of the gene, not at internal splice sites.

    We compare signal in windows around donor sites (exon→intron boundary)
    vs the gene-body average.

    Args:
        chrom: Chromosome name
        splice_sites_df: DataFrame with splice_type, position columns
        gene_start: Gene start coordinate
        gene_end: Gene end coordinate
        window: Window size for signal averaging (bp)
    """
    import pyBigWig

    results = {}

    donors = splice_sites_df.filter(
        pl.col("splice_type") == "donor"
    )["position"].unique().to_list()

    acceptors = splice_sites_df.filter(
        pl.col("splice_type") == "acceptor"
    )["position"].unique().to_list()

    for mark_name, url in [
        ("h3k36me3", BIGWIG_URLS["h3k36me3"]),
        ("h3k4me3", BIGWIG_URLS["h3k4me3"]),
    ]:
        log.info(f"  Opening {mark_name} (remote)...")
        bw = pyBigWig.open(url)

        def _window_signal(positions: list[int], w: int = window) -> np.ndarray:
            signals = []
            for pos in positions:
                start = max(0, pos - w)
                end = pos + w
                vals = bw.values(chrom, start, end)
                if vals is not None:
                    arr = np.array(vals, dtype=np.float64)
                    arr = np.nan_to_num(arr, nan=0.0)
                    signals.append(np.mean(arr))
                else:
                    signals.append(0.0)
            return np.array(signals)

        # Signal at splice sites
        donor_signal = _window_signal(donors)
        acceptor_signal = _window_signal(acceptors)

        # Gene body background (sample evenly-spaced positions)
        n_bg = min(50, (gene_end - gene_start) // 1000)
        bg_positions = np.linspace(gene_start + 500, gene_end - 500, n_bg).astype(int).tolist()
        bg_signal = _window_signal(bg_positions)

        bw.close()

        results[mark_name] = {
            "donor_mean": float(np.mean(donor_signal)) if len(donor_signal) > 0 else 0.0,
            "acceptor_mean": float(np.mean(acceptor_signal)) if len(acceptor_signal) > 0 else 0.0,
            "background_mean": float(np.mean(bg_signal)) if len(bg_signal) > 0 else 0.0,
            "n_donors": len(donors),
            "n_acceptors": len(acceptors),
        }

    return results


def print_epigenetic_results(results: dict) -> None:
    """Pretty-print epigenetic mark comparison."""
    for mark in ["h3k36me3", "h3k4me3"]:
        r = results[mark]
        label = "H3K36me3 (exon body mark)" if "36" in mark else "H3K4me3 (promoter mark)"
        print(f"\n  {label}:")
        print(f"    {'Region':<25s} {'Mean Signal':>12s}")
        print(f"    {'Donor sites (n={})'.format(r['n_donors']):<25s} "
              f"{r['donor_mean']:>12.3f}")
        print(f"    {'Acceptor sites (n={})'.format(r['n_acceptors']):<25s} "
              f"{r['acceptor_mean']:>12.3f}")
        print(f"    {'Gene body background':<25s} {r['background_mean']:>12.3f}")

        # Interpret
        splice_avg = (r["donor_mean"] + r["acceptor_mean"]) / 2
        bg = max(r["background_mean"], 1e-6)
        enrichment = splice_avg / bg
        if "36" in mark:
            print(f"    Splice-site/background enrichment: {enrichment:.2f}x")
            if enrichment > 1.2:
                print("    -> H3K36me3 enriched at splice sites (expected: exon marking)")
            else:
                print("    -> Weak enrichment (cell-type dependent — K562 is a leukemia line)")
        else:
            print(f"    Splice-site/background ratio: {enrichment:.2f}x")
            print("    -> H3K4me3 should peak at promoter, not internal splice sites")


# ═══════════════════════════════════════════════════════════════════════════
# Modality 3: RNA-seq Junction Evidence
# ═══════════════════════════════════════════════════════════════════════════

def explore_junction_evidence(splice_sites_df: pl.DataFrame) -> None:
    """Demonstrate how RNA-seq junction evidence works conceptually.

    Unlike conservation and epigenetic marks (available as genome-wide bigWig
    tracks), junction evidence comes from RNA-seq alignment — it's
    sample-specific and must be generated from BAM/SJ.out.tab files.

    This section explains the data format and how to integrate it, using
    TP53's known splice structure as an example.
    """
    # Count unique splice sites per type
    donors = splice_sites_df.filter(pl.col("splice_type") == "donor")
    acceptors = splice_sites_df.filter(pl.col("splice_type") == "acceptor")
    n_transcripts = splice_sites_df["transcript_id"].n_unique()

    print(f"\n  Gene splice structure:")
    print(f"    Transcripts: {n_transcripts}")
    print(f"    Unique donor positions:    {donors['position'].n_unique()}")
    print(f"    Unique acceptor positions:  {acceptors['position'].n_unique()}")

    print("""
  STAR SJ.out.tab format (tab-separated, no header):
  ┌────────┬──────────┬──────────┬────────┬────────┬──────┬───────────┬──────┬────────┐
  │ chrom  │ intron   │ intron   │ strand │ motif  │ anno │ n_unique  │ n_   │ max    │
  │        │ start    │ end      │ (0/1/2)│ (0-6)  │(0/1) │ _mappers  │ multi│ ovhang │
  └────────┴──────────┴──────────┴────────┴────────┴──────┴───────────┴──────┴────────┘

  Key columns for splice evidence:
  - intron_start, intron_end: define the junction (donor+1 to acceptor-1)
  - n_unique_mappers: uniquely-mapped reads spanning this junction
    (THIS IS THE SIGNAL — higher = more confident the junction is real)
  - annotated: 0 = novel junction, 1 = annotated in GTF
    (novel + high read count = candidate alternative splicing event)

  How to load (example):""")

    print("""    import polars as pl

    STAR_COLUMNS = [
        "chrom", "intron_start", "intron_end", "strand_code",
        "motif_code", "annotated", "n_unique", "n_multi", "max_overhang",
    ]

    junctions = pl.read_csv(
        "SJ.out.tab",
        separator="\\t",
        has_header=False,
        new_columns=STAR_COLUMNS,
    )

    # Filter to confident junctions
    confident = junctions.filter(pl.col("n_unique") >= 3)

    # Map junction boundaries to splice sites:
    #   donor_position    = intron_start - 1  (last exonic base)
    #   acceptor_position = intron_end + 1    (first exonic base)

    # For meta-layer features, junction read count becomes a feature:
    #   - log1p(n_unique) as continuous feature
    #   - Binary: has_junction_support (n_unique >= threshold)
    #   - Ratio: n_unique / max_junction_in_gene (normalized)""")

    print("""
  Public junction data sources:
  - GTEx (bulk RNA-seq, 54 tissues): https://gtexportal.org/home/downloads
    -> junction read counts per tissue (large files, ~GB each)
  - ENCODE (cell lines): search for "RNA-seq" + cell line + "alignments"
  - For TP53: any cancer RNA-seq dataset will have abundant junction reads
    (TP53 is the most mutated gene in human cancers)

  Integration path for meta-layer:
  1. Download tissue-specific SJ.out.tab files
  2. Parse and index by (chrom, intron_start, intron_end)
  3. Join to splice site positions → junction count per site
  4. Feature: log1p(n_unique), has_support, tissue_breadth (n_tissues)""")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feature Engineering Example 5: Multimodal Exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gene", default="TP53", help="Gene symbol (default: TP53)")
    parser.add_argument(
        "--model", default="openspliceai", choices=["openspliceai", "spliceai"],
        help="Base model for annotations (default: openspliceai)",
    )
    parser.add_argument(
        "--n-random", type=int, default=30,
        help="Number of random intronic control positions (default: 30)",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Conservation averaging half-window in bp (default: 10)",
    )
    parser.add_argument(
        "--skip-remote", action="store_true",
        help="Skip remote bigWig queries (show junction section only)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Feature Engineering Example 5: Multimodal Exploration")
    print("=" * 70)
    print(f"\nGene: {args.gene}")
    print(f"Model: {args.model}")
    print(f"Random control positions: {args.n_random}")

    # ── Load splice site annotations ────────────────────────────────────
    print(f"\nStep 0: Loading splice sites for {args.gene}...")
    t0 = time.time()
    splice_df = get_splice_sites(args.gene, args.model)
    dt = time.time() - t0
    print(f"  Loaded {splice_df.height} splice site annotations in {dt:.1f}s")
    print(f"  Columns: {splice_df.columns}")

    chrom = splice_df["chrom"][0]
    all_positions = splice_df["position"].unique().sort().to_list()
    gene_start = min(all_positions) - 1000
    gene_end = max(all_positions) + 1000
    print(f"  Chromosome: {chrom}")
    print(f"  Gene region: {gene_start:,} - {gene_end:,} ({(gene_end - gene_start):,} bp)")
    print(f"  Unique splice positions: {len(all_positions)}")

    # Sample random intronic control positions
    random_positions = sample_intronic_positions(
        chrom, gene_start, gene_end,
        set(all_positions), n=args.n_random,
    )
    print(f"  Random control positions: {len(random_positions)}")

    if args.skip_remote:
        print("\n  --skip-remote: Skipping conservation and epigenetic queries.")
    else:
        if not check_pybigwig():
            return 1

        # ── Modality 1: Conservation ────────────────────────────────────
        print("\n" + "=" * 70)
        print("Modality 1: Conservation Scores (PhyloP / PhastCons)")
        print("=" * 70)
        print("""
  WHY: Splice sites are among the most conserved elements in the genome.
  The GT donor and AG acceptor dinucleotides are nearly invariant across
  vertebrates. Strong conservation = functional constraint = real splice site.
  Weak conservation at a predicted splice site may indicate a false positive
  or a lineage-specific alternative event.
""")

        t0 = time.time()
        conservation = explore_conservation(
            chrom, all_positions, random_positions, window=args.window,
        )
        dt = time.time() - t0
        print_conservation_results(conservation)
        print(f"\n  (Fetched in {dt:.1f}s)")

        # ── Modality 2: Epigenetic marks ────────────────────────────────
        print("\n" + "=" * 70)
        print("Modality 2: Epigenetic Marks (H3K36me3 / H3K4me3)")
        print("=" * 70)
        print("""
  WHY: Histone modifications leave a chromatin signature of gene activity.
  - H3K36me3 is deposited on EXON BODIES during transcription. The
    spliceosome recruits SETD2, which methylates H3K36 as Pol II passes.
    Higher H3K36me3 at a splice site = the exon is actively used.
  - H3K4me3 marks active PROMOTERS. It's less relevant to internal splice
    sites but useful for detecting gene activity.

  Cell line: K562 (chronic myelogenous leukemia, ENCODE Tier 1).
  Note: Epigenetic signals are cell-type-specific! A splice site that
  shows weak H3K36me3 in K562 may be active in other tissues.
""")

        t0 = time.time()
        epigenetic = explore_epigenetic(
            chrom, splice_df, gene_start, gene_end, window=500,
        )
        dt = time.time() - t0
        print_epigenetic_results(epigenetic)
        print(f"\n  (Fetched in {dt:.1f}s)")

    # ── Modality 3: Junction evidence ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Modality 3: RNA-seq Junction Evidence")
    print("=" * 70)
    print("""
  WHY: Junction reads are DIRECT experimental evidence of splicing.
  Unlike conservation (evolutionary proxy) or histone marks (chromatin
  proxy), junction reads prove that a splice event actually occurs in a
  specific sample/tissue. This is the strongest single signal for
  confirming predicted splice sites.

  Unlike bigWig tracks, junction data is sample-specific and must be
  generated from RNA-seq alignments. Below we explain the data format
  and integration path.
""")
    explore_junction_evidence(splice_df)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary: How These Modalities Complement Base Predictions")
    print("=" * 70)
    print("""
  Base model (SpliceAI/OpenSpliceAI) predicts splice sites from DNA SEQUENCE
  alone. Each additional modality adds orthogonal evidence:

  ┌──────────────────┬────────────────────┬──────────────────────────────┐
  │ Modality         │ Signal Type        │ What It Tells Us             │
  ├──────────────────┼────────────────────┼──────────────────────────────┤
  │ Base scores      │ Sequence pattern   │ Is the GT/AG motif intact?   │
  │ Conservation     │ Evolutionary       │ Is this site under selection? │
  │ H3K36me3         │ Chromatin          │ Is the exon actively used?   │
  │ H3K4me3          │ Chromatin          │ Is the gene active?          │
  │ Junction reads   │ Experimental       │ Does splicing actually occur? │
  │ (future) RNA     │ Structure          │ Does local structure affect  │
  │  structure       │                    │ splice site accessibility?   │
  └──────────────────┴────────────────────┴──────────────────────────────┘

  Meta-layer fusion strategy:
  1. Extract per-position features from each modality
  2. Concatenate into a feature matrix (positions x features)
  3. Train a meta-model (gradient-boosted trees or small NN) that learns
     which modality combinations predict real splice sites
  4. Delta scores (meta - base) reveal sites that need additional evidence

  Next steps:
  -> examples/features/01-04: See the existing feature pipeline in action
  -> src/agentic_spliceai/splice_engine/features/: Add new modalities here
  -> foundation_models/: Deep embeddings as another feature source
""")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

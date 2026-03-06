"""Gene-level train/val/test splitting strategies for genomics ML.

Provides chromosome-holdout splitting with optional homology-aware filtering,
following established practices in splice site prediction (SpliceAI, OpenSpliceAI).

Key design principles:
  - **Split by gene, not by window/sample** — all windows from one gene go to the
    same split. Prevents information leakage from overlapping windows.
  - **Chromosome holdout** — biologically meaningful groups that prevent positional
    and linkage leakage.
  - **Paralog exclusion from test set** — test genes with homologs in training are
    removed, ensuring the model is evaluated on truly unseen gene families.
  - **Separate val and test sets** — val is used for early stopping (indirectly
    optimized), test is fully held out for final reporting.

Preset splits:
  - ``spliceai`` — SpliceAI's published split (Jaganathan et al., 2019)
  - ``even_odd`` — Even chromosomes train, odd test (simple)
  - ``custom`` — User-defined chromosome assignments

Usage::

    from agentic_spliceai.splice_engine.eval.splitting import (
        GeneSplit,
        build_gene_split,
        SPLIT_PRESETS,
    )

    # Using SpliceAI's preset
    split = build_gene_split(
        gene_chromosomes={"BRCA1": "chr17", "TP53": "chr17", ...},
        preset="spliceai",
        val_fraction=0.1,
        seed=42,
    )
    print(split.train_genes, split.val_genes, split.test_genes)

    # With homology filtering
    split = build_gene_split(
        gene_chromosomes=gene_chrom_map,
        preset="spliceai",
        val_fraction=0.1,
        gene_families=family_map,  # from homology.detect_gene_families_by_name()
        seed=42,
    )
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chromosome split presets
# ---------------------------------------------------------------------------

# SpliceAI (Jaganathan et al., 2019):
#   Train: chr2, 4, 6, 8, 10-22, X, Y
#   Test:  chr1, 3, 5, 7, 9
# Val is carved from training genes (10% random holdout for early stopping).
SPLICEAI_TRAIN_CHROMS = {
    "chr2", "chr4", "chr6", "chr8",
    "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
    "chr17", "chr18", "chr19", "chr20", "chr21", "chr22",
    "chrX", "chrY",
}
SPLICEAI_TEST_CHROMS = {"chr1", "chr3", "chr5", "chr7", "chr9"}


@dataclass
class ChromosomeSplitConfig:
    """Defines which chromosomes belong to train vs test."""

    train_chromosomes: Set[str]
    test_chromosomes: Set[str]
    description: str = ""


SPLIT_PRESETS: Dict[str, ChromosomeSplitConfig] = {
    "spliceai": ChromosomeSplitConfig(
        train_chromosomes=SPLICEAI_TRAIN_CHROMS,
        test_chromosomes=SPLICEAI_TEST_CHROMS,
        description=(
            "SpliceAI split (Jaganathan et al., 2019): "
            "train=even+chr10-22+X+Y, test=chr1,3,5,7,9"
        ),
    ),
    "even_odd": ChromosomeSplitConfig(
        train_chromosomes={f"chr{c}" for c in range(2, 23, 2)} | {"chrX", "chrY"},
        test_chromosomes={f"chr{c}" for c in range(1, 22, 2)},
        description="Even chromosomes (+ X, Y) for training, odd for testing",
    ),
}


# ---------------------------------------------------------------------------
# Gene split result
# ---------------------------------------------------------------------------


@dataclass
class GeneSplit:
    """Result of a gene-level train/val/test split.

    All gene IDs are stored as sets for O(1) membership testing.
    """

    train_genes: Set[str] = field(default_factory=set)
    val_genes: Set[str] = field(default_factory=set)
    test_genes: Set[str] = field(default_factory=set)

    # Genes removed from test due to paralogy with training genes
    test_paralogs_removed: Set[str] = field(default_factory=set)

    # Metadata
    preset: str = ""
    seed: int = 42
    val_fraction: float = 0.1

    @property
    def n_train(self) -> int:
        return len(self.train_genes)

    @property
    def n_val(self) -> int:
        return len(self.val_genes)

    @property
    def n_test(self) -> int:
        return len(self.test_genes)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    def summary(self) -> str:
        """Human-readable summary of the split."""
        lines = [
            f"Gene Split ({self.preset or 'custom'})",
            f"  Train: {self.n_train} genes",
            f"  Val:   {self.n_val} genes (from training chromosomes, {self.val_fraction:.0%} holdout)",
            f"  Test:  {self.n_test} genes",
        ]
        if self.test_paralogs_removed:
            lines.append(
                f"  Paralogs removed from test: {len(self.test_paralogs_removed)} genes"
            )
        lines.append(f"  Total: {self.n_total} genes")
        lines.append(f"  Seed:  {self.seed}")
        return "\n".join(lines)

    def validate(self) -> None:
        """Check that splits don't overlap."""
        train_val = self.train_genes & self.val_genes
        train_test = self.train_genes & self.test_genes
        val_test = self.val_genes & self.test_genes

        if train_val:
            raise ValueError(f"Train/val overlap: {len(train_val)} genes")
        if train_test:
            raise ValueError(f"Train/test overlap: {len(train_test)} genes")
        if val_test:
            raise ValueError(f"Val/test overlap: {len(val_test)} genes")

    def to_dict(self) -> Dict:
        """Serialize to dict (for JSON/YAML persistence)."""
        return {
            "train_genes": sorted(self.train_genes),
            "val_genes": sorted(self.val_genes),
            "test_genes": sorted(self.test_genes),
            "test_paralogs_removed": sorted(self.test_paralogs_removed),
            "preset": self.preset,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> GeneSplit:
        """Deserialize from dict."""
        return cls(
            train_genes=set(data["train_genes"]),
            val_genes=set(data["val_genes"]),
            test_genes=set(data["test_genes"]),
            test_paralogs_removed=set(data.get("test_paralogs_removed", [])),
            preset=data.get("preset", ""),
            seed=data.get("seed", 42),
            val_fraction=data.get("val_fraction", 0.1),
        )


# ---------------------------------------------------------------------------
# Split builders
# ---------------------------------------------------------------------------


def _normalize_chrom(chrom: str) -> str:
    """Normalize chromosome name to 'chr' prefix format."""
    if not chrom.startswith("chr"):
        return f"chr{chrom}"
    return chrom


def build_gene_split(
    gene_chromosomes: Dict[str, str],
    preset: str = "spliceai",
    val_fraction: float = 0.1,
    seed: int = 42,
    gene_families: Optional[Dict[str, str]] = None,
    exclude_test_paralogs: bool = True,
    custom_train_chroms: Optional[Set[str]] = None,
    custom_test_chroms: Optional[Set[str]] = None,
) -> GeneSplit:
    """Build a gene-level train/val/test split using chromosome holdout.

    Parameters
    ----------
    gene_chromosomes:
        Dict mapping gene_id (or gene_name) -> chromosome (e.g., "chr17").
        All genes to be split.
    preset:
        Split preset name: "spliceai", "even_odd", or "custom".
        If "custom", must provide ``custom_train_chroms`` and ``custom_test_chroms``.
    val_fraction:
        Fraction of training-chromosome genes to hold out for validation
        (early stopping). Drawn randomly with seed control.
    seed:
        Random seed for reproducible val split.
    gene_families:
        Optional dict mapping gene_id -> family_id. Used for homology-aware
        paralog exclusion from test set. Output of
        ``data.homology.detect_gene_families_by_name()`` or similar.
    exclude_test_paralogs:
        If True and ``gene_families`` is provided, remove test genes whose
        family has any member in the training set. Default True.
    custom_train_chroms:
        Train chromosomes (only used when preset="custom").
    custom_test_chroms:
        Test chromosomes (only used when preset="custom").

    Returns
    -------
    GeneSplit with non-overlapping train/val/test gene sets.
    """
    # Resolve chromosome split
    if preset == "custom":
        if custom_train_chroms is None or custom_test_chroms is None:
            raise ValueError("preset='custom' requires custom_train_chroms and custom_test_chroms")
        chrom_config = ChromosomeSplitConfig(
            train_chromosomes=custom_train_chroms,
            test_chromosomes=custom_test_chroms,
            description="Custom chromosome split",
        )
    elif preset in SPLIT_PRESETS:
        chrom_config = SPLIT_PRESETS[preset]
    else:
        raise ValueError(
            f"Unknown preset: {preset!r}. Available: {', '.join(SPLIT_PRESETS)} or 'custom'"
        )

    logger.info("Using split: %s", chrom_config.description)

    # Assign genes to train vs test based on chromosome
    train_pool: List[str] = []
    test_pool: List[str] = []
    unassigned: List[str] = []

    for gene_id, chrom in gene_chromosomes.items():
        chrom_norm = _normalize_chrom(chrom)
        if chrom_norm in chrom_config.train_chromosomes:
            train_pool.append(gene_id)
        elif chrom_norm in chrom_config.test_chromosomes:
            test_pool.append(gene_id)
        else:
            unassigned.append(gene_id)

    if unassigned:
        logger.warning(
            "%d genes on chromosomes not in train or test set (e.g., %s). "
            "Adding to training.",
            len(unassigned), unassigned[0] if unassigned else "?",
        )
        train_pool.extend(unassigned)

    logger.info(
        "Chromosome split: %d train-pool, %d test-pool",
        len(train_pool), len(test_pool),
    )

    # Homology-aware paralog exclusion from test set
    paralogs_removed: Set[str] = set()
    if gene_families and exclude_test_paralogs:
        paralogs_removed = _exclude_test_paralogs(
            train_genes=set(train_pool),
            test_genes=set(test_pool),
            gene_families=gene_families,
        )
        test_pool = [g for g in test_pool if g not in paralogs_removed]
        logger.info(
            "Paralog exclusion: removed %d test genes, %d remaining",
            len(paralogs_removed), len(test_pool),
        )

    # Split train pool into train + val
    rng = random.Random(seed)
    train_pool_shuffled = sorted(train_pool)  # Sort first for determinism
    rng.shuffle(train_pool_shuffled)

    n_val = max(1, int(len(train_pool_shuffled) * val_fraction))
    val_genes = set(train_pool_shuffled[:n_val])
    train_genes = set(train_pool_shuffled[n_val:])

    split = GeneSplit(
        train_genes=train_genes,
        val_genes=val_genes,
        test_genes=set(test_pool),
        test_paralogs_removed=paralogs_removed,
        preset=preset,
        seed=seed,
        val_fraction=val_fraction,
    )

    split.validate()

    logger.info(
        "Final split: %d train, %d val, %d test",
        split.n_train, split.n_val, split.n_test,
    )

    return split


def _exclude_test_paralogs(
    train_genes: Set[str],
    test_genes: Set[str],
    gene_families: Dict[str, str],
) -> Set[str]:
    """Find test genes whose family has any member in the training set.

    Parameters
    ----------
    train_genes:
        Set of training gene IDs.
    test_genes:
        Set of test gene IDs.
    gene_families:
        Dict mapping gene_id -> family_id.

    Returns
    -------
    Set of test gene IDs to remove (they have train-set paralogs).
    """
    # Build family -> member sets
    from ..data.homology import get_paralog_groups
    groups = get_paralog_groups(gene_families)

    # Find families that have at least one member in training
    train_families: Set[str] = set()
    for gene in train_genes:
        if gene in gene_families:
            train_families.add(gene_families[gene])

    # Flag test genes in those families
    to_remove: Set[str] = set()
    for gene in test_genes:
        if gene in gene_families and gene_families[gene] in train_families:
            to_remove.add(gene)

    return to_remove


# ---------------------------------------------------------------------------
# Convenience: build gene_chromosomes dict from common data sources
# ---------------------------------------------------------------------------


def gene_chromosomes_from_dataframe(
    df: "polars.DataFrame | pandas.DataFrame",
    gene_col: str = "gene_id",
    chrom_col: str = "chrom",
) -> Dict[str, str]:
    """Extract gene->chromosome mapping from a DataFrame.

    Works with both polars and pandas DataFrames. If the DataFrame has
    ``seqname`` instead of ``chrom``, it will be used automatically.

    Parameters
    ----------
    df:
        DataFrame with gene and chromosome columns.
    gene_col:
        Column name for gene identifiers.
    chrom_col:
        Column name for chromosome.

    Returns
    -------
    Dict mapping gene_id -> chromosome string.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            # Try alternate column names
            if chrom_col not in df.columns and "seqname" in df.columns:
                chrom_col = "seqname"
            if gene_col not in df.columns and "gene_name" in df.columns:
                gene_col = "gene_name"
            return dict(zip(
                df[gene_col].to_list(),
                df[chrom_col].to_list(),
            ))
    except ImportError:
        pass

    import pandas as pd
    if isinstance(df, pd.DataFrame):
        if chrom_col not in df.columns and "seqname" in df.columns:
            chrom_col = "seqname"
        if gene_col not in df.columns and "gene_name" in df.columns:
            gene_col = "gene_name"
        return dict(zip(df[gene_col], df[chrom_col]))

    raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def gene_chromosomes_from_gtf(
    gtf_path: str,
    gene_type: str = "protein_coding",
) -> Dict[str, str]:
    """Load gene->chromosome mapping from a GTF file.

    Parameters
    ----------
    gtf_path:
        Path to GTF annotation file.
    gene_type:
        Filter to this gene biotype. Pass None to include all.

    Returns
    -------
    Dict mapping gene_name -> chromosome.
    """
    from ..base_layer.data.genomic_extraction import extract_gene_annotations

    df = extract_gene_annotations(str(gtf_path), verbosity=0)

    if gene_type and "gene_type" in df.columns:
        import polars as pl
        df = df.filter(pl.col("gene_type") == gene_type)

    return gene_chromosomes_from_dataframe(df, gene_col="gene_name", chrom_col="seqname")

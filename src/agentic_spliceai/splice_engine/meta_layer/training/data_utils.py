"""
Shared data utilities for meta-layer training scripts.

Provides loading, splitting, and feature selection functions used across
M1-M4 training scripts (``examples/meta_layer/``).  Keeps reusable logic
in ``src/`` so that example scripts stay thin.

Usage::

    from agentic_spliceai.splice_engine.meta_layer.training.data_utils import (
        load_analysis_sequences,
        get_gene_split,
        split_dataframe,
        get_feature_columns,
        resolve_input_dir,
        MODALITY_COLUMNS,
        EXCLUDE_COLS,
        LABEL_COL,
        LABEL_ENCODING,
        LABEL_DECODING,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from agentic_spliceai.splice_engine.eval.splitting import (
    GeneSplit,
    build_gene_split,
    gene_chromosomes_from_dataframe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_COL = "splice_type"

LABEL_ENCODING = {"donor": 0, "acceptor": 1, "neither": 2, "": 2}
LABEL_DECODING = {0: "donor", 1: "acceptor", 2: "neither"}

# Columns that must NEVER be used as features.
LEAKAGE_COLS = {
    "splice_type",
    "pred_type",
    "true_position",
    "predicted_position",
    "is_correct",
    "error_type",
}

METADATA_COLS = {
    "gene_id",
    "gene_name",
    "transcript_id",
    "gene_type",
    "chrom",
    "strand",
    "position",
    "absolute_position",
    "window_start",
    "window_end",
    "transcript_count",
    "gene_start",
    "gene_end",
}

NON_NUMERIC_COLS = {"sequence"}

EXCLUDE_COLS = LEAKAGE_COLS | METADATA_COLS | NON_NUMERIC_COLS | {LABEL_COL}

# Authoritative modality → column mapping.  Covers all 10 modalities
# (9 active + fm_embeddings which may be absent in current parquets).
MODALITY_COLUMNS: Dict[str, List[str]] = {
    "base_scores": [
        "donor_score", "acceptor_score", "neither_score",
        "context_score_m2", "context_score_m1", "context_score_p1", "context_score_p2",
        "relative_donor_probability", "splice_probability", "donor_acceptor_diff",
        "splice_neither_diff", "donor_acceptor_logodds", "splice_neither_logodds",
        "probability_entropy", "context_neighbor_mean", "context_asymmetry", "context_max",
        "donor_diff_m1", "donor_diff_m2", "donor_diff_p1", "donor_diff_p2",
        "donor_surge_ratio", "donor_is_local_peak", "donor_weighted_context",
        "donor_peak_height_ratio", "donor_second_derivative", "donor_signal_strength",
        "donor_context_diff_ratio",
        "acceptor_diff_m1", "acceptor_diff_m2", "acceptor_diff_p1", "acceptor_diff_p2",
        "acceptor_surge_ratio", "acceptor_is_local_peak", "acceptor_weighted_context",
        "acceptor_peak_height_ratio", "acceptor_second_derivative", "acceptor_signal_strength",
        "acceptor_context_diff_ratio",
        "donor_acceptor_peak_ratio", "type_signal_difference",
        "score_difference_ratio", "signal_strength_ratio",
    ],
    "genomic": [
        "relative_gene_position", "distance_to_gene_start",
        "distance_to_gene_end", "gc_content",
    ],
    "conservation": [
        "phylop_score", "phylop_context_mean", "phylop_context_max", "phylop_context_std",
        "phastcons_score", "phastcons_context_mean", "phastcons_context_max",
        "phastcons_context_std", "conservation_contrast",
    ],
    "epigenetic": [
        "h3k36me3_max_across_tissues", "h3k36me3_mean_across_tissues",
        "h3k36me3_tissue_breadth", "h3k36me3_variance",
        "h3k36me3_context_mean", "h3k36me3_exon_intron_ratio",
        "h3k4me3_max_across_tissues", "h3k4me3_mean_across_tissues",
        "h3k4me3_tissue_breadth", "h3k4me3_variance",
        "h3k4me3_context_mean", "h3k4me3_exon_intron_ratio",
    ],
    "junction": [
        "junction_log1p", "junction_has_support", "junction_n_partners",
        "junction_max_reads", "junction_entropy", "junction_is_annotated",
        "junction_tissue_breadth", "junction_tissue_max", "junction_tissue_mean",
        "junction_tissue_variance", "junction_psi", "junction_psi_variance",
    ],
    "rbp_eclip": [
        "rbp_n_bound", "rbp_max_signal", "rbp_max_neg_log10_pvalue",
        "rbp_has_splice_regulator", "rbp_n_sr_proteins", "rbp_n_hnrnps",
        "rbp_cell_line_breadth", "rbp_mean_signal",
    ],
    "chrom_access": [
        "atac_max_across_tissues", "atac_mean_across_tissues",
        "atac_tissue_breadth", "atac_variance",
        "atac_context_mean", "atac_has_peak",
        "dnase_max_across_tissues", "dnase_mean_across_tissues",
        "dnase_tissue_breadth", "dnase_variance",
        "dnase_context_mean", "dnase_has_peak",
    ],
    "fm_embeddings": [
        "fm_pca_1", "fm_pca_2", "fm_pca_3", "fm_pca_4",
        "fm_pca_5", "fm_pca_6", "fm_embedding_norm", "fm_local_gradient",
    ],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def resolve_input_dir(
    input_dir: Optional[Path] = None,
    base_model: str = "openspliceai",
) -> Path:
    """Resolve analysis_sequences directory from explicit path or registry.

    Parameters
    ----------
    input_dir:
        Explicit path.  If None, auto-resolve from the model registry.
    base_model:
        Base model name for registry lookup (default: openspliceai).

    Returns
    -------
    Path to the analysis_sequences directory.

    Raises
    ------
    FileNotFoundError
        If the directory cannot be found.
    """
    if input_dir is not None:
        return Path(input_dir)

    from agentic_spliceai.splice_engine.resources import get_model_resources

    resources = get_model_resources(base_model)
    registry = resources.get_registry()
    resolved = registry.get_base_model_eval_dir(base_model) / "analysis_sequences"
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Cannot find analysis_sequences directory for {base_model}. "
        f"Tried: {resolved}\n"
        f"Run feature engineering first:\n"
        f"  python examples/features/06_multimodal_genome_workflow.py --chromosomes all"
    )


def load_analysis_sequences(
    input_dir: Path,
    chromosomes: List[str],
) -> pl.DataFrame:
    """Load analysis_sequences parquets for specified chromosomes.

    Parameters
    ----------
    input_dir:
        Directory containing ``analysis_sequences_{chrom}.parquet`` files.
    chromosomes:
        List of chromosome names (e.g., ``["chr1", "chr2"]``).

    Returns
    -------
    Concatenated Polars DataFrame with schema-aligned columns.
    """
    frames = []
    for chrom in chromosomes:
        path = input_dir / f"analysis_sequences_{chrom}.parquet"
        if not path.exists():
            # Try TSV fallback
            path = input_dir / f"analysis_sequences_{chrom}.tsv"
            if not path.exists():
                logger.warning("No data for %s at %s", chrom, input_dir)
                continue
            frames.append(pl.read_csv(path, separator="\t"))
        else:
            frames.append(pl.read_parquet(path))
        logger.info("Loaded %s: %d positions", chrom, frames[-1].height)

    if not frames:
        raise FileNotFoundError(
            f"No analysis_sequences found in {input_dir} "
            f"for chromosomes: {chromosomes}"
        )

    # Align schemas: use column intersection (different chromosomes may have
    # been generated with different modality configs)
    if len(frames) > 1:
        common_cols = set(frames[0].columns)
        for f in frames[1:]:
            common_cols &= set(f.columns)
        max_cols = max(len(f.columns) for f in frames)
        if len(common_cols) < max_cols:
            dropped = max_cols - len(common_cols)
            logger.info(
                "Schema alignment: keeping %d common columns (dropped %d)",
                len(common_cols), dropped,
            )
            ordered_cols = [c for c in frames[0].columns if c in common_cols]
            frames = [f.select(ordered_cols) for f in frames]

    # Ensure consistent column order before concat (parquets generated at
    # different times may have the same columns in different order).
    canonical_order = frames[0].columns
    frames = [f.select(canonical_order) for f in frames]

    return pl.concat(frames)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def get_gene_split(
    df: pl.DataFrame,
    preset: str = "spliceai",
    val_fraction: float = 0.1,
    seed: int = 42,
    custom_train_chroms: Optional[Set[str]] = None,
    custom_test_chroms: Optional[Set[str]] = None,
) -> GeneSplit:
    """Build a gene-level train/val/test split from a DataFrame.

    Thin wrapper that extracts gene-chromosome mapping and delegates to
    :func:`build_gene_split`.

    Parameters
    ----------
    df:
        DataFrame with ``gene_id`` (or ``gene_name``) and ``chrom`` columns.
    preset:
        Split preset: ``"spliceai"`` (default), ``"even_odd"``,
        ``"balanced"``, or ``"custom"``.
    val_fraction:
        Fraction of training genes held out for validation.
    seed:
        Random seed for val split.
    custom_train_chroms:
        Train chromosomes (only when preset="custom").
    custom_test_chroms:
        Test chromosomes (only when preset="custom").

    Returns
    -------
    GeneSplit with non-overlapping train/val/test gene sets.
    """
    gene_chromosomes = gene_chromosomes_from_dataframe(df)
    return build_gene_split(
        gene_chromosomes,
        preset=preset,
        val_fraction=val_fraction,
        seed=seed,
        custom_train_chroms=custom_train_chroms,
        custom_test_chroms=custom_test_chroms,
    )


def split_dataframe(
    df: pl.DataFrame,
    gene_split: GeneSplit,
    gene_col: str = "gene_id",
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a DataFrame into train/val/test using a GeneSplit.

    Parameters
    ----------
    df:
        DataFrame with a gene identifier column.
    gene_split:
        Result from :func:`get_gene_split` or :func:`build_gene_split`.
    gene_col:
        Column name for gene identifiers.

    Returns
    -------
    (df_train, df_val, df_test) — three non-overlapping DataFrames.
    """
    # Fall back to gene_name if gene_col is missing
    if gene_col not in df.columns and "gene_name" in df.columns:
        gene_col = "gene_name"

    genes = df[gene_col]

    df_train = df.filter(genes.is_in(list(gene_split.train_genes)))
    df_val = df.filter(genes.is_in(list(gene_split.val_genes)))
    df_test = df.filter(genes.is_in(list(gene_split.test_genes)))

    logger.info(
        "Split: train=%d, val=%d, test=%d positions",
        df_train.height, df_val.height, df_test.height,
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

_NUMERIC_TYPES = {
    pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32,
}


def get_feature_columns(
    df: pl.DataFrame,
    exclude_modalities: Optional[List[str]] = None,
    extra_exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Select numeric feature columns, optionally excluding entire modalities.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    exclude_modalities:
        Modality names to exclude (e.g., ``["junction"]`` for M3).
        Columns belonging to these modalities are dropped from the
        feature set.
    extra_exclude:
        Additional column names to exclude beyond the standard set.

    Returns
    -------
    Ordered list of feature column names.
    """
    excluded = set(EXCLUDE_COLS)
    if extra_exclude:
        excluded |= extra_exclude

    # Drop all columns belonging to excluded modalities
    if exclude_modalities:
        for mod_name in exclude_modalities:
            if mod_name in MODALITY_COLUMNS:
                excluded |= set(MODALITY_COLUMNS[mod_name])
            else:
                logger.warning("Unknown modality '%s' in exclude_modalities", mod_name)

    feature_cols = [
        c for c in df.columns
        if df[c].dtype in _NUMERIC_TYPES and c not in excluded
    ]

    logger.info("Selected %d feature columns", len(feature_cols))
    return feature_cols


def prepare_features_and_labels(
    df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_modalities: Optional[List[str]] = None,
    label_col: str = LABEL_COL,
    label_encoding: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X and label vector y from a DataFrame.

    Parameters
    ----------
    df:
        DataFrame with features and labels.
    feature_cols:
        Explicit feature columns. If None, auto-detected via
        :func:`get_feature_columns`.
    exclude_modalities:
        Passed to :func:`get_feature_columns` when ``feature_cols`` is None.
    label_col:
        Column name for the label.
    label_encoding:
        Mapping from label strings to integers.

    Returns
    -------
    (X, y, feature_names):
        X is float32 [n, d], y is int [n], feature_names is list of str.
    """
    if label_encoding is None:
        label_encoding = LABEL_ENCODING

    if feature_cols is None:
        feature_cols = get_feature_columns(df, exclude_modalities=exclude_modalities)

    if not feature_cols:
        raise ValueError(f"No valid feature columns found. Available: {df.columns}")

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")

    labels = df[label_col].to_list()
    y = np.array([label_encoding.get(str(l).lower(), 2) for l in labels])

    return X, y, feature_cols

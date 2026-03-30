"""Foundation model embeddings modality — scalar summary features.

Extracts per-position scalar features from pre-computed foundation model
embeddings (Evo2, SpliceBERT, HyenaDNA, etc.). This is **Tier 1**: dimensionality-
reduced summary statistics that plug directly into the existing XGBoost/tabular
meta-layer pipeline.

The modality is a **reader**, not a compute step — embeddings must be pre-extracted
on a GPU pod using the ``foundation_models`` sub-project (e.g.,
``07a_direct_shard_splice_predictor.py``), then indexed into per-chromosome
parquet files aligned with the feature pipeline's sampled positions.

Scalar features include:
- **PCA components** (top-k principal components fit on training chromosomes)
- **Embedding norm** (L2 magnitude — correlates with model confidence)
- **Local embedding gradient** (L2 norm of position-to-position difference)

All features are **label-agnostic** — they do not use splice site annotations,
avoiding any risk of label leakage. The PCA captures dominant variation modes
in embedding space; the meta-layer model learns which modes correlate with
splicing.

PCA transformer is **fit on training chromosomes only** to prevent data leakage,
saved as a ``.npz`` artifact, and loaded at transform time.

Centroid cosine similarity features (donor/acceptor) are available but disabled
by default — they use ground truth labels to compute centroids, which is
technically supervised feature engineering (not leakage with proper train/test
split, but redundant with base model scores that already capture splice-likeness).

Requires pre-extracted embeddings and PCA artifacts. If the embedding directory
does not exist or chromosomes are missing, all columns are filled with NaN
(graceful degradation, same pattern as conservation/epigenetic for GRCh37).

See Also
--------
examples/features/docs/fm-embeddings-tutorial.md
    Full tutorial on foundation model embeddings, PCA fitting, and
    feature interpretation.
foundation_models/docs/results/splice-site-prediction-results.md
    SpliceBERT frozen-head classifier results (AUPRC=0.832).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


# ── Supported foundation models ──────────────────────────────────────

FM_MODEL_DIMS: dict[str, int] = {
    "evo2_7b": 4096,
    "evo2_40b": 8192,
    "splicebert": 512,
    "hyenadna": 256,
    "nucleotide_transformer": 1024,
}

DEFAULT_FOUNDATION_MODEL = "evo2_7b"
DEFAULT_PCA_COMPONENTS = 6


@dataclass
class FMEmbeddingsConfig(ModalityConfig):
    """Configuration for the foundation model embeddings modality.

    Attributes
    ----------
    foundation_model : str
        Foundation model name. Must be a key in ``FM_MODEL_DIMS``.
        Default: 'evo2_7b' (4096-dim per position).
    scalar_dir : Path or None
        **Recommended.** Directory containing pre-computed scalar parquets
        from ``07_streaming_fm_scalars.py``. Expected structure:
        ``{scalar_dir}/fm_scalars_{chrom}.parquet`` with columns:
        ``chrom``, ``position``, ``fm_pca_1``..``fm_pca_k``,
        ``fm_embedding_norm``, ``fm_local_gradient``.
        When set, the modality reads scalars directly (no PCA projection).
    embedding_dir : Path or None
        Directory containing per-chromosome embedding parquet files.
        Expected structure: ``{embedding_dir}/embeddings_{chrom}.parquet``
        with columns: ``chrom``, ``position``, ``embedding`` (list[f32]).
        Used only when ``scalar_dir`` is not set.
        If neither scalar_dir nor embedding_dir is set, degrades gracefully.
    pca_components : int
        Number of PCA components to include. Default: 6.
    pca_artifacts_path : Path or None
        Path to ``.npz`` file containing pre-fit PCA artifacts:
        ``components`` (k x d), ``mean`` (d,).
        Required only when using ``embedding_dir`` (not ``scalar_dir``).
    include_cosine_centroids : bool
        Include cosine similarity to donor/acceptor embedding centroids.
        Requires centroids in the PCA artifacts file. Default: False
        (disabled — uses ground truth labels, redundant with base scores).
    include_gradient : bool
        Include local embedding gradient (L2 norm of difference with
        neighbors within the same gene). Default: True.
    """

    foundation_model: str = DEFAULT_FOUNDATION_MODEL
    scalar_dir: Optional[Path] = None
    embedding_dir: Optional[Path] = None
    pca_components: int = DEFAULT_PCA_COMPONENTS
    pca_artifacts_path: Optional[Path] = None
    include_cosine_centroids: bool = False
    include_gradient: bool = True


class FMEmbeddingsModality(Modality):
    """Scalar features derived from foundation model per-position embeddings.

    Tier 1: dimensionality-reduced summary statistics suitable for
    XGBoost and other tabular meta-layer models.

    Embeddings must be pre-extracted on a GPU pod. The ``transform()``
    method reads pre-indexed per-chromosome parquet files and applies
    a pre-fit PCA projection + summary statistics.

    If embeddings or PCA artifacts are not available, all columns are
    filled with NaN (graceful degradation).
    """

    def __init__(self, config: FMEmbeddingsConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: FMEmbeddingsConfig = self.config  # type: ignore[assignment]
        self._pca_loaded = False
        self._pca_components: Optional[np.ndarray] = None  # (k, d)
        self._pca_mean: Optional[np.ndarray] = None  # (d,)
        self._donor_centroid: Optional[np.ndarray] = None  # (d,)
        self._acceptor_centroid: Optional[np.ndarray] = None  # (d,)

    @property
    def meta(self) -> ModalityMeta:
        cols = self._compute_output_columns()
        return ModalityMeta(
            name="fm_embeddings",
            version="0.1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset({"chrom", "position", "gene_id"}),
            optional_inputs=frozenset(),
            description=(
                f"Scalar features from {self._cfg.foundation_model} "
                f"embeddings (PCA + norm + centroid similarity + gradient)."
            ),
        )

    def _compute_output_columns(self) -> list[str]:
        """Compute output column names based on config."""
        cols: list[str] = []

        # PCA components
        for i in range(1, self._cfg.pca_components + 1):
            cols.append(f"fm_pca_{i}")

        # Embedding norm (always included)
        cols.append("fm_embedding_norm")

        # Centroid cosine similarities
        if self._cfg.include_cosine_centroids:
            cols.append("fm_donor_cosine_sim")
            cols.append("fm_acceptor_cosine_sim")

        # Local embedding gradient
        if self._cfg.include_gradient:
            cols.append("fm_local_gradient")

        return cols

    @classmethod
    def default_config(cls) -> FMEmbeddingsConfig:
        return FMEmbeddingsConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)

        if self._cfg.foundation_model not in FM_MODEL_DIMS:
            errors.append(
                f"Unknown foundation model: '{self._cfg.foundation_model}'. "
                f"Available: {list(FM_MODEL_DIMS.keys())}"
            )

        if self._cfg.pca_components < 1:
            errors.append(
                f"pca_components must be >= 1, got {self._cfg.pca_components}"
            )

        # Warn (not error) if data is missing — graceful degradation
        if self._cfg.embedding_dir is not None:
            emb_dir = Path(self._cfg.embedding_dir)
            if not emb_dir.exists():
                logger.warning(
                    "Embedding directory '%s' does not exist. "
                    "All fm_embeddings features will be NaN. "
                    "Extract embeddings on a GPU pod first.",
                    emb_dir,
                )

        if self._cfg.pca_artifacts_path is not None:
            pca_path = Path(self._cfg.pca_artifacts_path)
            if not pca_path.exists():
                logger.warning(
                    "PCA artifacts '%s' not found. "
                    "PCA and centroid features will be NaN. "
                    "Run the PCA fitting script on training chromosomes first.",
                    pca_path,
                )

        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add foundation model embedding scalar features to the DataFrame."""
        n_positions = df.height
        logger.info(
            "FM embeddings modality: %d positions, model=%s, "
            "pca_components=%d, centroids=%s, gradient=%s",
            n_positions,
            self._cfg.foundation_model,
            self._cfg.pca_components,
            self._cfg.include_cosine_centroids,
            self._cfg.include_gradient,
        )

        # Mode 1: Pre-computed scalar parquets (recommended, from 07_streaming_fm_scalars.py)
        if self._cfg.scalar_dir is not None and Path(self._cfg.scalar_dir).exists():
            return self._load_precomputed_scalars(df)

        # Mode 2: Raw embedding parquets + PCA projection
        if self._cfg.embedding_dir is not None and Path(self._cfg.embedding_dir).exists():
            return self._transform_from_embeddings(df)

        # No data available — graceful degradation
        logger.warning(
            "No scalar_dir or embedding_dir available. "
            "Filling all fm_embeddings columns with NaN."
        )
        return self._fill_nan(df)

    def _load_precomputed_scalars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Load pre-computed scalar features and join onto the DataFrame.

        Reads per-chromosome ``fm_scalars_{chrom}.parquet`` files produced
        by ``07_streaming_fm_scalars.py`` and left-joins on (chrom, position).
        """
        scalar_dir = Path(self._cfg.scalar_dir)  # type: ignore[arg-type]
        output_cols = self._compute_output_columns()

        all_scalars = []
        chroms = df["chrom"].unique().to_list()
        for chrom in chroms:
            candidates = [
                scalar_dir / f"fm_scalars_{chrom}.parquet",
                scalar_dir / f"fm_scalars_chr{chrom}.parquet"
                if not str(chrom).startswith("chr") else None,
                scalar_dir / f"fm_scalars_{str(chrom).replace('chr', '')}.parquet"
                if str(chrom).startswith("chr") else None,
            ]
            candidates = [c for c in candidates if c is not None]

            loaded = False
            for candidate in candidates:
                if candidate.exists():
                    scalar_df = pl.read_parquet(candidate)
                    all_scalars.append(scalar_df)
                    loaded = True
                    logger.info("  Loaded pre-computed scalars: %s", candidate)
                    break

            if not loaded:
                logger.debug("No scalar parquet for %s. Will fill NaN.", chrom)

        if not all_scalars:
            logger.warning("No pre-computed scalar files found. Filling NaN.")
            return self._fill_nan(df)

        scalars = pl.concat(all_scalars)

        # Left join on (chrom, position)
        join_cols = [c for c in output_cols if c in scalars.columns]
        if not join_cols:
            logger.warning(
                "Scalar parquets have no expected columns. Filling NaN."
            )
            return self._fill_nan(df)

        scalars_subset = scalars.select(["chrom", "position"] + join_cols)
        df = df.join(scalars_subset, on=["chrom", "position"], how="left")

        # Fill any missing output columns with NaN
        for col in output_cols:
            if col not in df.columns:
                df = df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias(col)
                )

        n_matched = df.select(join_cols[0]).drop_nulls().height
        logger.info(
            "FM embeddings modality: %d/%d positions matched from scalars, "
            "added %d columns",
            n_matched, df.height, len(output_cols),
        )
        return df

    def _transform_from_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform from raw embedding parquets (Mode 2)."""
        # Load PCA artifacts (once)
        self._ensure_pca_loaded()

        # Load embeddings for chromosomes present in the DataFrame
        embeddings = self._load_embeddings(df)

        if embeddings is None:
            logger.warning(
                "No embeddings matched. Filling all fm_embeddings columns with NaN."
            )
            return self._fill_nan(df)

        # Compute features
        df = self._compute_features(df, embeddings)

        logger.info(
            "FM embeddings modality: added %d columns",
            len(self.meta.output_columns),
        )
        return df

    # ── Feature computation ──────────────────────────────────────────

    def _compute_features(
        self,
        df: pl.DataFrame,
        embeddings: np.ndarray,
    ) -> pl.DataFrame:
        """Compute all scalar features from the embedding matrix.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.
        embeddings : np.ndarray
            Shape (n_positions, hidden_dim). Rows aligned with df.
            NaN rows where no embedding was found.
        """
        n = df.height
        hidden_dim = embeddings.shape[1]

        # ── PCA projection ───────────────────────────────────────────
        pca_values = np.full((n, self._cfg.pca_components), np.nan)
        if self._pca_components is not None and self._pca_mean is not None:
            # Mask out rows with any NaN (missing embeddings)
            valid_mask = ~np.any(np.isnan(embeddings), axis=1)
            if np.any(valid_mask):
                centered = embeddings[valid_mask] - self._pca_mean
                projected = centered @ self._pca_components.T  # (n_valid, k)
                pca_values[valid_mask] = projected

        for i in range(self._cfg.pca_components):
            df = df.with_columns(
                pl.Series(f"fm_pca_{i + 1}", pca_values[:, i], dtype=pl.Float64)
            )

        # ── Embedding norm ───────────────────────────────────────────
        norms = np.full(n, np.nan)
        valid_mask = ~np.any(np.isnan(embeddings), axis=1)
        if np.any(valid_mask):
            norms[valid_mask] = np.linalg.norm(embeddings[valid_mask], axis=1)

        df = df.with_columns(
            pl.Series("fm_embedding_norm", norms, dtype=pl.Float64)
        )

        # ── Centroid cosine similarity ───────────────────────────────
        if self._cfg.include_cosine_centroids:
            donor_sim = np.full(n, np.nan)
            acceptor_sim = np.full(n, np.nan)

            if (
                self._donor_centroid is not None
                and self._acceptor_centroid is not None
                and np.any(valid_mask)
            ):
                valid_emb = embeddings[valid_mask]
                valid_norms = norms[valid_mask]

                # Avoid division by zero
                safe_norms = np.where(valid_norms > 0, valid_norms, 1.0)

                # Donor cosine similarity
                donor_dot = valid_emb @ self._donor_centroid
                donor_norm = np.linalg.norm(self._donor_centroid)
                if donor_norm > 0:
                    donor_sim[valid_mask] = donor_dot / (safe_norms * donor_norm)

                # Acceptor cosine similarity
                acceptor_dot = valid_emb @ self._acceptor_centroid
                acceptor_norm = np.linalg.norm(self._acceptor_centroid)
                if acceptor_norm > 0:
                    acceptor_sim[valid_mask] = acceptor_dot / (safe_norms * acceptor_norm)

            df = df.with_columns([
                pl.Series("fm_donor_cosine_sim", donor_sim, dtype=pl.Float64),
                pl.Series("fm_acceptor_cosine_sim", acceptor_sim, dtype=pl.Float64),
            ])

        # ── Local embedding gradient ─────────────────────────────────
        if self._cfg.include_gradient:
            gradient = self._compute_local_gradient(df, embeddings)
            df = df.with_columns(
                pl.Series("fm_local_gradient", gradient, dtype=pl.Float64)
            )

        return df

    def _compute_local_gradient(
        self,
        df: pl.DataFrame,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute local embedding gradient within each gene.

        For each position, compute the L2 norm of the difference between
        its embedding and the mean of its immediate neighbors (within the
        same gene). Uses ``.over('gene_id')`` logic to prevent cross-gene
        contamination.

        Returns NaN for positions at gene boundaries or with missing embeddings.
        """
        n = df.height
        gradient = np.full(n, np.nan)

        valid_mask = ~np.any(np.isnan(embeddings), axis=1)

        # Group by gene_id to avoid cross-gene leakage
        gene_ids = df["gene_id"].to_numpy()
        unique_genes = np.unique(gene_ids)

        for gene in unique_genes:
            gene_mask = gene_ids == gene
            gene_indices = np.where(gene_mask)[0]

            if len(gene_indices) < 3:
                continue

            for i in range(1, len(gene_indices) - 1):
                idx = gene_indices[i]
                prev_idx = gene_indices[i - 1]
                next_idx = gene_indices[i + 1]

                if not (valid_mask[idx] and valid_mask[prev_idx] and valid_mask[next_idx]):
                    continue

                neighbor_mean = (embeddings[prev_idx] + embeddings[next_idx]) / 2.0
                diff = embeddings[idx] - neighbor_mean
                gradient[idx] = np.linalg.norm(diff)

        return gradient

    # ── Data loading ─────────────────────────────────────────────────

    def _load_embeddings(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Load pre-extracted embeddings and align with DataFrame positions.

        Reads per-chromosome parquet files from ``embedding_dir``, joins
        on ``(chrom, position)``, and returns an aligned embedding matrix.

        Returns
        -------
        np.ndarray or None
            Shape (n_positions, hidden_dim). NaN for positions without
            embeddings. None if no embeddings could be loaded.
        """
        emb_dir = Path(self._cfg.embedding_dir)  # type: ignore[arg-type]
        hidden_dim = FM_MODEL_DIMS.get(self._cfg.foundation_model, 512)
        n = df.height

        embeddings = np.full((n, hidden_dim), np.nan)
        any_loaded = False

        chroms = df["chrom"].unique().to_list()
        for chrom in chroms:
            # Try multiple naming conventions
            candidates = [
                emb_dir / f"embeddings_{chrom}.parquet",
                emb_dir / f"embeddings_chr{chrom}.parquet" if not str(chrom).startswith("chr") else None,
                emb_dir / f"embeddings_{str(chrom).replace('chr', '')}.parquet" if str(chrom).startswith("chr") else None,
            ]
            candidates = [c for c in candidates if c is not None]

            emb_path = None
            for candidate in candidates:
                if candidate.exists():
                    emb_path = candidate
                    break

            if emb_path is None:
                logger.debug(
                    "No embedding file for chromosome '%s'. Skipping.", chrom
                )
                continue

            logger.info("  Loading embeddings from %s", emb_path)
            chrom_emb_df = pl.read_parquet(emb_path)

            # Join on (chrom, position)
            chrom_mask = df["chrom"] == chrom
            chrom_indices = np.where(chrom_mask.to_numpy())[0]
            positions = df.filter(chrom_mask)["position"].to_numpy()

            # Build position → embedding lookup
            emb_positions = chrom_emb_df["position"].to_numpy()
            emb_vectors = chrom_emb_df["embedding"].to_numpy()

            pos_to_idx = {int(p): i for i, p in enumerate(emb_positions)}

            n_matched = 0
            for df_row_idx, pos in zip(chrom_indices, positions):
                emb_idx = pos_to_idx.get(int(pos))
                if emb_idx is not None:
                    vec = emb_vectors[emb_idx]
                    if isinstance(vec, (list, np.ndarray)):
                        arr = np.asarray(vec, dtype=np.float64)
                        if len(arr) == hidden_dim:
                            embeddings[df_row_idx] = arr
                            n_matched += 1

            if n_matched > 0:
                any_loaded = True

            logger.info(
                "  %s: %d/%d positions matched",
                chrom, n_matched, len(positions),
            )

        return embeddings if any_loaded else None

    def _ensure_pca_loaded(self) -> None:
        """Load PCA artifacts from disk (once)."""
        if self._pca_loaded:
            return

        self._pca_loaded = True

        if self._cfg.pca_artifacts_path is None:
            logger.info(
                "No PCA artifacts path configured. "
                "PCA features will use NaN; norm and gradient still computed."
            )
            return

        pca_path = Path(self._cfg.pca_artifacts_path)
        if not pca_path.exists():
            logger.warning(
                "PCA artifacts not found at '%s'. "
                "PCA and centroid features will be NaN.",
                pca_path,
            )
            return

        logger.info("Loading PCA artifacts from %s", pca_path)
        data = np.load(pca_path)

        if "components" in data and "mean" in data:
            self._pca_components = data["components"]  # (k, d)
            self._pca_mean = data["mean"]  # (d,)
            k = self._pca_components.shape[0]
            if k < self._cfg.pca_components:
                logger.warning(
                    "PCA artifacts have %d components but config requests %d. "
                    "Using %d components; remaining will be NaN.",
                    k, self._cfg.pca_components, k,
                )
            logger.info(
                "PCA loaded: %d components, hidden_dim=%d",
                k, self._pca_components.shape[1],
            )

        if "donor_centroid" in data:
            self._donor_centroid = data["donor_centroid"]
        if "acceptor_centroid" in data:
            self._acceptor_centroid = data["acceptor_centroid"]

        if self._donor_centroid is not None and self._acceptor_centroid is not None:
            logger.info("Splice centroids loaded (donor + acceptor)")
        elif self._cfg.include_cosine_centroids:
            logger.warning(
                "Centroid features requested but centroids not found in "
                "PCA artifacts. Cosine similarity will be NaN."
            )

    # ── Helpers ───────────────────────────────────────────────────────

    def _fill_nan(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill all output columns with NaN (graceful degradation)."""
        for col_name in self._compute_output_columns():
            df = df.with_columns(
                pl.lit(None).cast(pl.Float64).alias(col_name)
            )
        return df

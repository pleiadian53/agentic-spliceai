"""Junction modality — RNA-seq splice junction read evidence.

Extracts junction-level features from STAR SJ.out.tab files or
pre-aggregated multi-tissue junction tables. Features are attributed
to splice site boundary positions (donor and acceptor) via a sparse
join — most genomic positions are not junction boundaries and receive
zero values.

Supports two input formats:

- **STAR SJ.out.tab** (single sample): 9-column headerless TSV from
  STAR aligner. Converted to donor/acceptor positions internally.
- **Aggregated TSV** (multi-tissue): Pre-processed from GTEx/recount3
  with per-tissue read counts. Enables tissue-breadth and variance
  features.

And two aggregation modes:

- **summarized** (default): Cross-tissue summary statistics (~12 cols).
- **detailed**: Per-tissue features for exploratory analysis.

This modality is **label-agnostic** — it produces the same feature
columns regardless of downstream usage. The meta-layer training config
decides whether junction columns are features (M2 model) or held-out
targets (M3 model).

See Also
--------
examples/features/docs/junction-reads-tutorial.md
    Full tutorial on junction data preparation and interpretation.
docs/meta_layer/predicting_induced_splice_sites/
    Design docs on alternative/induced splice site prediction.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


# ── STAR SJ.out.tab column definitions ──────────────────────────────
# STAR produces a headerless 9-column TSV. Column order is fixed.
STAR_SJ_COLUMNS = [
    "chrom",          # chromosome
    "intron_start",   # 1-based first intronic base
    "intron_end",     # 1-based last intronic base
    "strand",         # 0=undefined, 1=+, 2=-
    "intron_motif",   # 0=non-canonical, 1=GT/AG, 2=CT/AC, 3=GC/AG, etc.
    "annotated",      # 0=novel, 1=annotated in GTF
    "unique_reads",   # uniquely-mapping split reads
    "multi_reads",    # multi-mapping split reads
    "max_overhang",   # maximum overhang of split reads
]

STAR_STRAND_MAP = {0: ".", 1: "+", 2: "-"}


@dataclass
class JunctionConfig(ModalityConfig):
    """Configuration for the junction modality.

    Attributes
    ----------
    base_model : str
        Base model name for build/resource resolution.
    junction_data_path : Path or None
        Path to junction data file (STAR SJ.out.tab or aggregated TSV).
        If None, auto-resolved from the genomic registry.
    min_support : int
        Minimum unique reads to count a junction as supported.
        Junctions below this threshold are filtered out.
    aggregation : str
        Aggregation mode: 'summarized' (cross-tissue stats) or
        'detailed' (per-tissue columns).
    breadth_threshold : int
        Minimum reads per tissue to count toward tissue_breadth.
    include_psi : bool
        Whether to compute simplified PSI columns. PSI measures how
        dominant a junction is among competing alternatives at the
        same boundary position.
    """

    base_model: str = "openspliceai"
    junction_data_path: Optional[Path] = None
    min_support: int = 3
    aggregation: str = "summarized"
    breadth_threshold: int = 3
    include_psi: bool = True


class JunctionModality(Modality):
    """Extract junction-level features from RNA-seq data.

    Junction features are **sparse**: they are attributed to splice site
    boundary positions (donor and acceptor). Positions that are not at a
    junction boundary receive 0.0 for count-based features and NaN for
    ratio-based features (PSI).

    Each junction in the data creates features at BOTH its donor and
    acceptor positions. When multiple junctions share a boundary position
    (competing donors or acceptors), features aggregate across all
    junctions anchored at that position.
    """

    def __init__(self, config: JunctionConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: JunctionConfig = self.config  # type: ignore[assignment]
        self._junction_index: Optional[pl.DataFrame] = None
        self._build: Optional[str] = None
        self._data_format: Optional[str] = None  # "star" or "aggregated"

    @property
    def meta(self) -> ModalityMeta:
        cols = self._compute_output_columns()
        return ModalityMeta(
            name="junction",
            version="0.1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset({"chrom", "position"}),
            optional_inputs=frozenset({"strand"}),
            description="RNA-seq junction read evidence (STAR/GTEx/recount3).",
        )

    @classmethod
    def default_config(cls) -> JunctionConfig:
        return JunctionConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)
        # Try to resolve junction data path — warn but don't fail
        try:
            path = self._resolve_junction_path()
            if path is not None and not path.exists():
                errors.append(
                    f"Junction data file not found: {path}. "
                    f"Pipeline will run with zero-filled junction features."
                )
                errors.clear()  # Downgrade to warning, not error
                logger.warning(
                    "Junction data not found at %s — features will be zero-filled.",
                    path,
                )
        except Exception:
            # No junction data available — graceful degradation
            logger.warning(
                "Could not resolve junction data path. "
                "Junction features will be zero-filled."
            )
        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Join junction features onto the prediction DataFrame.

        Builds a pre-aggregated junction index keyed by (chrom, position),
        then performs a Polars left-join. Positions not at junction
        boundaries get 0.0 for count-based features, NaN for PSI.
        """
        junction_index = self._get_junction_index()

        if junction_index is None or junction_index.height == 0:
            return self._fill_defaults(df)

        # Left join: keep all prediction rows
        output_cols = list(self._compute_output_columns())
        df = df.join(junction_index, on=["chrom", "position"], how="left")

        # Fill nulls: 0.0 for count-based, NaN stays for PSI
        count_cols = [c for c in output_cols if not c.startswith("junction_psi")]
        psi_cols = [c for c in output_cols if c.startswith("junction_psi")]

        fill_exprs = [pl.col(c).fill_null(0.0) for c in count_cols if c in df.columns]
        if fill_exprs:
            df = df.with_columns(fill_exprs)

        # Ensure all declared output columns exist
        for col in output_cols:
            if col not in df.columns:
                if col in psi_cols:
                    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
                else:
                    df = df.with_columns(pl.lit(0.0).alias(col))

        return df

    # ------------------------------------------------------------------
    # Output column computation
    # ------------------------------------------------------------------

    def _compute_output_columns(self) -> list[str]:
        """Compute output columns based on config."""
        cols = [
            "junction_log1p",
            "junction_has_support",
            "junction_n_partners",
            "junction_max_reads",
            "junction_entropy",
            "junction_is_annotated",
        ]

        # Tissue-level columns (always present; zero for single-sample data)
        cols.extend([
            "junction_tissue_breadth",
            "junction_tissue_max",
            "junction_tissue_mean",
            "junction_tissue_variance",
        ])

        if self._cfg.include_psi:
            cols.extend([
                "junction_psi",
                "junction_psi_variance",
            ])

        return cols

    # ------------------------------------------------------------------
    # Junction index construction
    # ------------------------------------------------------------------

    def _get_junction_index(self) -> Optional[pl.DataFrame]:
        """Load and build the junction index (cached)."""
        if self._junction_index is not None:
            return self._junction_index

        raw = self._load_junctions()
        if raw is None:
            return None

        self._junction_index = self._build_junction_index(raw)
        return self._junction_index

    def _load_junctions(self) -> Optional[pl.DataFrame]:
        """Load junction data, auto-detecting format."""
        path = self._resolve_junction_path()
        if path is None or not path.exists():
            logger.warning("No junction data available. Features will be zero-filled.")
            return None

        logger.info("Loading junction data from %s", path)

        # Auto-detect format: STAR SJ.out.tab has no header (9 columns)
        # Try reading first line to check
        with open(path) as f:
            first_line = f.readline().strip()

        # STAR SJ.out.tab: no header, all fields numeric except chrom
        fields = first_line.split("\t")
        has_header = not fields[1].isdigit() if len(fields) > 1 else False

        if not has_header and len(fields) == len(STAR_SJ_COLUMNS):
            self._data_format = "star"
            return self._load_star_sj(path)
        else:
            self._data_format = "aggregated"
            return self._load_aggregated(path)

    def _load_star_sj(self, path: Path) -> pl.DataFrame:
        """Load STAR SJ.out.tab (headerless 9-column TSV)."""
        df = pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=STAR_SJ_COLUMNS,
        )

        # Convert intron coords to boundary positions
        # STAR: intron_start = 1-based first intronic base = donor_pos + 1
        #        intron_end = 1-based last intronic base = acceptor_pos - 1
        df = df.with_columns(
            (pl.col("intron_start") - 1).alias("donor_pos"),
            (pl.col("intron_end") + 1).alias("acceptor_pos"),
        )

        # Map strand encoding
        df = df.with_columns(
            pl.col("strand")
            .replace_strict(STAR_STRAND_MAP, default=".")
            .alias("strand"),
        )

        # Filter by minimum support
        df = df.filter(pl.col("unique_reads") >= self._cfg.min_support)

        # Normalize chrom prefix
        df = self._normalize_chrom(df)

        logger.info(
            "Loaded %d junctions from STAR SJ.out.tab (after min_support=%d filter)",
            df.height,
            self._cfg.min_support,
        )
        return df

    def _load_aggregated(self, path: Path) -> pl.DataFrame:
        """Load pre-aggregated multi-tissue junction table."""
        df = pl.read_csv(path, separator="\t")

        # Normalize column names
        rename_map = {
            "reads": "unique_reads",
            "score": "unique_reads",
            "count": "unique_reads",
            "support_count": "unique_reads",
            "start": "donor_pos",
            "end": "acceptor_pos",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename({old: new})

        # Validate required columns
        required = {"chrom", "donor_pos", "acceptor_pos"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Aggregated junction file missing columns: {missing}. "
                f"Expected: chrom, donor_pos, acceptor_pos, [unique_reads, "
                f"tissue, annotated, strand]"
            )

        # Default unique_reads to 1 if not present
        if "unique_reads" not in df.columns:
            df = df.with_columns(pl.lit(1).alias("unique_reads"))

        # Filter by minimum support
        df = df.filter(pl.col("unique_reads") >= self._cfg.min_support)

        # Normalize chrom prefix
        df = self._normalize_chrom(df)

        has_tissue = "tissue" in df.columns
        logger.info(
            "Loaded %d junction records (aggregated format, multi_tissue=%s)",
            df.height,
            has_tissue,
        )
        return df

    def _build_junction_index(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Build a per-position junction feature index.

        "Melts" each junction into two rows (donor_pos, acceptor_pos),
        then aggregates features per (chrom, position).
        """
        has_tissue = "tissue" in raw.columns
        has_annotated = "annotated" in raw.columns

        # Melt: each junction → two boundary rows
        # Donor side
        donor_df = raw.select(
            pl.col("chrom"),
            pl.col("donor_pos").alias("position"),
            pl.col("acceptor_pos").alias("partner_pos"),
            pl.lit("donor").alias("boundary_type"),
            pl.col("unique_reads"),
            *([pl.col("annotated")] if has_annotated else [pl.lit(0).alias("annotated")]),
            *([pl.col("tissue")] if has_tissue else []),
        )

        # Acceptor side
        acceptor_df = raw.select(
            pl.col("chrom"),
            pl.col("acceptor_pos").alias("position"),
            pl.col("donor_pos").alias("partner_pos"),
            pl.lit("acceptor").alias("boundary_type"),
            pl.col("unique_reads"),
            *([pl.col("annotated")] if has_annotated else [pl.lit(0).alias("annotated")]),
            *([pl.col("tissue")] if has_tissue else []),
        )

        melted = pl.concat([donor_df, acceptor_df])

        if has_tissue:
            return self._aggregate_multi_tissue(melted)
        else:
            return self._aggregate_single_sample(melted)

    def _aggregate_single_sample(self, melted: pl.DataFrame) -> pl.DataFrame:
        """Aggregate junction features for single-sample data."""
        grouped = melted.group_by(["chrom", "position"]).agg([
            # Total reads across all junctions at this position
            pl.col("unique_reads").sum().alias("_total_reads"),
            # Max reads from any single junction
            pl.col("unique_reads").max().alias("junction_max_reads"),
            # Number of distinct partner positions
            pl.col("partner_pos").n_unique().alias("junction_n_partners"),
            # Is any junction annotated?
            pl.col("annotated").max().alias("_is_annotated"),
            # Read counts per partner (for entropy + PSI)
            pl.col("unique_reads").alias("_reads_list"),
        ])

        # Compute derived features
        grouped = grouped.with_columns(
            pl.col("_total_reads").log1p().alias("junction_log1p"),
            pl.lit(1.0).alias("junction_has_support"),
            pl.col("_is_annotated").cast(pl.Float64).alias("junction_is_annotated"),
            pl.col("junction_max_reads").cast(pl.Float64),
            pl.col("junction_n_partners").cast(pl.Float64),
        )

        # Entropy across partner read distribution
        entropy_vals = self._compute_entropy(grouped["_reads_list"])
        grouped = grouped.with_columns(
            pl.Series("junction_entropy", entropy_vals, dtype=pl.Float64),
        )

        # PSI: simplified position-level PSI
        if self._cfg.include_psi:
            psi_vals = self._compute_psi(grouped["_reads_list"])
            grouped = grouped.with_columns(
                pl.Series("junction_psi", psi_vals, dtype=pl.Float64),
                pl.lit(None).cast(pl.Float64).alias("junction_psi_variance"),
            )

        # Single-sample: tissue columns are zero
        grouped = grouped.with_columns(
            pl.lit(0.0).alias("junction_tissue_breadth"),
            pl.col("junction_max_reads").alias("junction_tissue_max"),
            pl.col("_total_reads").cast(pl.Float64).alias("junction_tissue_mean"),
            pl.lit(0.0).alias("junction_tissue_variance"),
        )

        # Select output columns
        output_cols = ["chrom", "position"] + list(self._compute_output_columns())
        return grouped.select([c for c in output_cols if c in grouped.columns])

    def _aggregate_multi_tissue(self, melted: pl.DataFrame) -> pl.DataFrame:
        """Aggregate junction features across tissues.

        First aggregates per (chrom, position, tissue) to get per-tissue
        totals, then computes cross-tissue summary statistics.
        """
        # Per-tissue aggregation
        per_tissue = melted.group_by(["chrom", "position", "tissue"]).agg([
            pl.col("unique_reads").sum().alias("tissue_reads"),
            pl.col("partner_pos").n_unique().alias("tissue_n_partners"),
            pl.col("annotated").max().alias("_is_annotated"),
            pl.col("unique_reads").alias("_reads_list"),
        ])

        # Cross-tissue aggregation per position
        grouped = per_tissue.group_by(["chrom", "position"]).agg([
            # Total reads across all tissues
            pl.col("tissue_reads").sum().alias("_total_reads"),
            # Max reads from any single tissue
            pl.col("tissue_reads").max().alias("junction_tissue_max"),
            # Mean reads across tissues
            pl.col("tissue_reads").mean().alias("junction_tissue_mean"),
            # Variance across tissues
            pl.col("tissue_reads").var().alias("junction_tissue_variance"),
            # Tissue breadth: tissues with reads >= breadth_threshold
            (pl.col("tissue_reads") >= self._cfg.breadth_threshold)
            .sum()
            .alias("junction_tissue_breadth"),
            # Max n_partners across tissues
            pl.col("tissue_n_partners").max().alias("junction_n_partners"),
            # Is any junction annotated?
            pl.col("_is_annotated").max().alias("_is_annotated"),
            # Collect per-tissue read lists for entropy/PSI
            pl.col("_reads_list").alias("_nested_reads"),
        ])

        # Compute derived features
        grouped = grouped.with_columns(
            pl.col("_total_reads").log1p().alias("junction_log1p"),
            pl.lit(1.0).alias("junction_has_support"),
            pl.col("_is_annotated").cast(pl.Float64).alias("junction_is_annotated"),
            pl.col("junction_tissue_max").cast(pl.Float64),
            pl.col("junction_tissue_mean").cast(pl.Float64),
            pl.col("junction_tissue_variance").fill_null(0.0).cast(pl.Float64),
            pl.col("junction_tissue_breadth").cast(pl.Float64),
            pl.col("junction_n_partners").cast(pl.Float64),
        )

        # Max reads from any single junction (across all tissues)
        grouped = grouped.with_columns(
            pl.col("junction_tissue_max").alias("junction_max_reads"),
        )

        # Entropy: flatten nested reads lists, compute entropy
        entropy_vals = self._compute_entropy_from_nested(grouped["_nested_reads"])
        grouped = grouped.with_columns(
            pl.Series("junction_entropy", entropy_vals, dtype=pl.Float64),
        )

        # PSI across tissues
        if self._cfg.include_psi:
            psi_vals, psi_var_vals = self._compute_psi_multi_tissue(
                grouped["_nested_reads"]
            )
            grouped = grouped.with_columns(
                pl.Series("junction_psi", psi_vals, dtype=pl.Float64),
                pl.Series("junction_psi_variance", psi_var_vals, dtype=pl.Float64),
            )

        # Select output columns
        output_cols = ["chrom", "position"] + list(self._compute_output_columns())
        return grouped.select([c for c in output_cols if c in grouped.columns])

    # ------------------------------------------------------------------
    # Feature computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy(reads_lists: pl.Series) -> list[float]:
        """Compute Shannon entropy of read distribution across partners.

        High entropy = many competing junctions with similar support.
        Zero entropy = single junction or one dominant junction.
        """
        results = []
        for reads_list in reads_lists.to_list():
            if reads_list is None or len(reads_list) <= 1:
                results.append(0.0)
                continue
            reads = np.array(reads_list, dtype=np.float64)
            total = reads.sum()
            if total == 0:
                results.append(0.0)
                continue
            probs = reads / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            results.append(float(entropy))
        return results

    @staticmethod
    def _compute_entropy_from_nested(nested_reads: pl.Series) -> list[float]:
        """Compute entropy from nested list-of-lists (multi-tissue).

        Flattens all per-tissue read lists into a single read vector
        per position, then computes entropy.
        """
        results = []
        for nested in nested_reads.to_list():
            if nested is None:
                results.append(0.0)
                continue
            # Flatten nested lists
            flat = []
            for sublist in nested:
                if sublist is not None:
                    if isinstance(sublist, (list, np.ndarray)):
                        flat.extend(sublist)
                    else:
                        flat.append(sublist)
            if len(flat) <= 1:
                results.append(0.0)
                continue
            reads = np.array(flat, dtype=np.float64)
            total = reads.sum()
            if total == 0:
                results.append(0.0)
                continue
            probs = reads / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            results.append(float(entropy))
        return results

    @staticmethod
    def _compute_psi(reads_lists: pl.Series) -> list[Optional[float]]:
        """Compute simplified position-level PSI for single-sample data.

        PSI = max_reads / total_reads at a position. Measures how
        dominant the strongest junction is among competing alternatives.

        Returns NaN when only one junction (PSI is undefined without
        competition).
        """
        results: list[Optional[float]] = []
        for reads_list in reads_lists.to_list():
            if reads_list is None or len(reads_list) <= 1:
                results.append(None)
                continue
            reads = np.array(reads_list, dtype=np.float64)
            total = reads.sum()
            if total == 0:
                results.append(None)
                continue
            results.append(float(reads.max() / total))
        return results

    @staticmethod
    def _compute_psi_multi_tissue(
        nested_reads: pl.Series,
    ) -> tuple[list[Optional[float]], list[Optional[float]]]:
        """Compute PSI mean and variance across tissues.

        For each tissue, PSI = max_reads / total_reads.
        Returns (mean_psi, var_psi) across tissues.
        """
        psi_means: list[Optional[float]] = []
        psi_vars: list[Optional[float]] = []

        for nested in nested_reads.to_list():
            if nested is None:
                psi_means.append(None)
                psi_vars.append(None)
                continue

            tissue_psis = []
            for sublist in nested:
                if sublist is None:
                    continue
                reads = sublist if isinstance(sublist, list) else [sublist]
                if len(reads) <= 1:
                    continue
                arr = np.array(reads, dtype=np.float64)
                total = arr.sum()
                if total > 0:
                    tissue_psis.append(float(arr.max() / total))

            if not tissue_psis:
                psi_means.append(None)
                psi_vars.append(None)
            else:
                psi_arr = np.array(tissue_psis)
                psi_means.append(float(np.mean(psi_arr)))
                psi_vars.append(
                    float(np.var(psi_arr)) if len(psi_arr) > 1 else None
                )

        return psi_means, psi_vars

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _fill_defaults(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill all junction columns with default values (no data available)."""
        output_cols = self._compute_output_columns()
        for col in output_cols:
            if col.startswith("junction_psi"):
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
            else:
                df = df.with_columns(pl.lit(0.0).alias(col))
        return df

    def _normalize_chrom(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize chromosome names to match the build convention.

        Ensures junction data chrom names match the input DataFrame's
        expected format. Uses the build's chr_prefix convention.
        """
        build = self._resolve_build()

        # GRCh38 / MANE uses chr prefix; GRCh37 / Ensembl does not
        uses_chr = build in ("GRCh38", "GRCh38_MANE")

        if uses_chr:
            # Ensure chr prefix
            df = df.with_columns(
                pl.when(pl.col("chrom").str.starts_with("chr"))
                .then(pl.col("chrom"))
                .otherwise(pl.concat_str([pl.lit("chr"), pl.col("chrom")]))
                .alias("chrom")
            )
        else:
            # Strip chr prefix
            df = df.with_columns(
                pl.col("chrom").str.replace("^chr", "").alias("chrom")
            )

        return df

    def _resolve_build(self) -> str:
        """Resolve genomic build from base_model (cached)."""
        if self._build is not None:
            return self._build

        from agentic_spliceai.splice_engine.resources import get_model_resources

        resources = get_model_resources(self._cfg.base_model)
        self._build = resources.build
        return self._build

    def _resolve_junction_path(self) -> Optional[Path]:
        """Resolve junction data file path.

        Resolution order:
        1. Explicit config path (junction_data_path)
        2. Registry auto-resolution (junctions.tsv in build dir)
        """
        if self._cfg.junction_data_path is not None:
            return Path(self._cfg.junction_data_path)

        # Auto-resolve from registry
        try:
            from agentic_spliceai.splice_engine.resources import get_model_resources

            resources = get_model_resources(self._cfg.base_model)
            registry = resources.get_registry()
            path = registry.resolve("junctions")
            return path
        except Exception:
            return None

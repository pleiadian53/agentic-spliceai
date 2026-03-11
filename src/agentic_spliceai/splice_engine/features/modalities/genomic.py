"""Genomic context modality — positional and compositional features.

Lightweight features computable from position metadata and optionally
from the DNA sequence column (if the sequence modality runs first).
"""

import logging
from dataclasses import dataclass
from typing import List, Set

import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


@dataclass
class GenomicContextConfig(ModalityConfig):
    """Configuration for the genomic context modality.

    Attributes
    ----------
    gc_window : int
        Window size (in bp) for GC content calculation.
        Only used if ``sequence`` column is available.
    include_dinucleotides : bool
        If True, compute CpG density from the sequence column.
    """

    gc_window: int = 100
    include_dinucleotides: bool = False


class GenomicContextModality(Modality):
    """Add positional and compositional genomic features.

    Always produces: relative_gene_position, distance_to_gene_start,
    distance_to_gene_end.

    Conditionally produces (when ``sequence`` column is available):
    gc_content, cpg_density (if include_dinucleotides=True).
    """

    def __init__(self, config: GenomicContextConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: GenomicContextConfig = self.config  # type: ignore[assignment]

    @property
    def meta(self) -> ModalityMeta:
        cols = [
            "relative_gene_position",
            "distance_to_gene_start",
            "distance_to_gene_end",
        ]
        optional = set()
        # Sequence-dependent columns are optional
        optional.add("sequence")
        cols.append("gc_content")
        if self._cfg.include_dinucleotides:
            cols.append("cpg_density")

        return ModalityMeta(
            name="genomic",
            version="1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset(
                {"position", "gene_start", "gene_end"}
            ),
            optional_inputs=frozenset(optional),
            description="Positional and compositional genomic features.",
        )

    @classmethod
    def default_config(cls) -> GenomicContextConfig:
        return GenomicContextConfig()

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add genomic context features to the DataFrame."""
        gene_len = pl.col("gene_end") - pl.col("gene_start")

        # Genomic relative position (always left-to-right on reference)
        genomic_rel = (
            (pl.col("position") - pl.col("gene_start")).cast(pl.Float64)
            / gene_len.cast(pl.Float64).clip(lower_bound=1)
        )

        # Transcriptomic relative position: 0.0 = 5' (TSS), 1.0 = 3' (TES)
        # + strand: same as genomic. - strand: reversed.
        if "strand" in df.columns:
            transcriptomic_rel = (
                pl.when(pl.col("strand") == "-")
                .then(1.0 - genomic_rel)
                .otherwise(genomic_rel)
            )
        else:
            transcriptomic_rel = genomic_rel

        df = df.with_columns(
            # Transcriptomic coordinate (0.0 = 5'/TSS, 1.0 = 3'/TES)
            transcriptomic_rel.alias("relative_gene_position"),
            # Absolute distances to gene boundaries (genomic, not strand-corrected)
            (pl.col("position") - pl.col("gene_start")).alias(
                "distance_to_gene_start"
            ),
            (pl.col("gene_end") - pl.col("position")).alias(
                "distance_to_gene_end"
            ),
        )

        # Sequence-dependent features
        if "sequence" in df.columns:
            df = self._add_sequence_features(df)
        else:
            # Add placeholder columns so downstream schema is consistent
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("gc_content"))
            if self._cfg.include_dinucleotides:
                df = df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("cpg_density")
                )

        return df

    def _add_sequence_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute GC content and optionally CpG density from sequences."""
        sequences = df["sequence"].to_list()
        w = self._cfg.gc_window

        gc_values: list[float | None] = []
        cpg_values: list[float | None] = []

        for seq in sequences:
            if seq is None or len(seq) == 0:
                gc_values.append(None)
                if self._cfg.include_dinucleotides:
                    cpg_values.append(None)
                continue

            # Extract central window for GC calculation
            center = len(seq) // 2
            start = max(0, center - w // 2)
            end = min(len(seq), center + w // 2)
            window = seq[start:end].upper()

            # GC content
            if len(window) > 0:
                gc = (window.count("G") + window.count("C")) / len(window)
            else:
                gc = None
            gc_values.append(gc)

            # CpG density
            if self._cfg.include_dinucleotides:
                if len(window) > 1:
                    cpg_count = sum(
                        1 for i in range(len(window) - 1) if window[i:i + 2] == "CG"
                    )
                    cpg_values.append(cpg_count / (len(window) - 1))
                else:
                    cpg_values.append(None)

        df = df.with_columns(pl.Series("gc_content", gc_values, dtype=pl.Float64))

        if self._cfg.include_dinucleotides:
            df = df.with_columns(
                pl.Series("cpg_density", cpg_values, dtype=pl.Float64)
            )

        return df

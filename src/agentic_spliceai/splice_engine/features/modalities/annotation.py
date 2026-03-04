"""Annotation modality — ground truth splice site labels.

Joins known donor/acceptor positions from pre-extracted splice site
annotations onto the prediction DataFrame, adding ``splice_type``
and optionally ``transcript_id`` and ``transcript_count`` columns.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


@dataclass
class AnnotationConfig(ModalityConfig):
    """Configuration for the annotation modality.

    Attributes
    ----------
    splice_sites_path : Path or None
        Path to ``splice_sites_enhanced.tsv``. If None, auto-resolved
        from the genomic registry based on base_model.
    base_model : str
        Base model name for auto-resolving paths (e.g., 'openspliceai').
    include_transcript_info : bool
        Whether to add transcript_id and transcript_count columns.
    """

    splice_sites_path: Optional[Path] = None
    base_model: str = "openspliceai"
    include_transcript_info: bool = True


class AnnotationModality(Modality):
    """Add ground truth splice_type labels by joining with annotations.

    Matches predictions to known splice sites by (chrom, position, strand).
    Positions not matching a known splice site get splice_type=''.
    """

    def __init__(self, config: AnnotationConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: AnnotationConfig = self.config  # type: ignore[assignment]
        self._splice_sites: Optional[pl.DataFrame] = None

    @property
    def meta(self) -> ModalityMeta:
        cols = ["splice_type"]
        if self._cfg.include_transcript_info:
            cols.extend(["transcript_id", "transcript_count"])
        return ModalityMeta(
            name="annotation",
            version="1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset({"chrom", "position", "strand"}),
            description="Ground truth splice site labels from GTF annotations.",
        )

    @classmethod
    def default_config(cls) -> AnnotationConfig:
        return AnnotationConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)
        # Ensure we can load annotations
        try:
            self._load_splice_sites()
        except Exception as e:
            errors.append(f"Cannot load splice site annotations: {e}")
        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Join splice site annotations onto the prediction DataFrame."""
        ss = self._load_splice_sites()

        # Build join keys: chrom + position + strand
        # Keep only the columns we need from annotations
        join_cols = ["chrom", "position", "strand", "splice_type"]
        if self._cfg.include_transcript_info:
            if "transcript_id" in ss.columns:
                join_cols.append("transcript_id")

        ss_subset = ss.select([c for c in join_cols if c in ss.columns])

        # Aggregate to one row per (chrom, position, strand) — a position
        # may appear in multiple transcripts
        agg_exprs: list[pl.Expr] = [pl.col("splice_type").first()]
        if self._cfg.include_transcript_info and "transcript_id" in ss_subset.columns:
            agg_exprs.extend([
                pl.col("transcript_id").first(),
                pl.col("transcript_id").n_unique().alias("transcript_count"),
            ])

        ss_agg = ss_subset.group_by(["chrom", "position", "strand"]).agg(agg_exprs)

        # Left join: keep all prediction rows, fill unmatched with defaults
        df = df.join(ss_agg, on=["chrom", "position", "strand"], how="left")

        # Fill nulls for positions without known splice sites
        df = df.with_columns(pl.col("splice_type").fill_null(""))

        if self._cfg.include_transcript_info:
            if "transcript_id" in df.columns:
                df = df.with_columns(pl.col("transcript_id").fill_null(""))
            else:
                df = df.with_columns(pl.lit("").alias("transcript_id"))

            if "transcript_count" in df.columns:
                df = df.with_columns(pl.col("transcript_count").fill_null(0))
            else:
                df = df.with_columns(pl.lit(0).alias("transcript_count"))

        return df

    def _load_splice_sites(self) -> pl.DataFrame:
        """Load splice site annotations (cached after first load)."""
        if self._splice_sites is not None:
            return self._splice_sites

        path = self._resolve_splice_sites_path()
        logger.info("Loading splice site annotations from %s", path)
        self._splice_sites = pl.read_csv(path, separator="\t")
        logger.info(
            "Loaded %d splice site annotations", self._splice_sites.height
        )
        return self._splice_sites

    def _resolve_splice_sites_path(self) -> Path:
        """Resolve the path to splice_sites_enhanced.tsv."""
        if self._cfg.splice_sites_path is not None:
            path = Path(self._cfg.splice_sites_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Splice sites file not found: {path}"
                )
            return path

        # Auto-resolve from registry
        from agentic_spliceai.splice_engine.resources import get_model_resources

        resources = get_model_resources(self._cfg.base_model)
        annotations_dir = resources.get_annotations_dir()
        path = annotations_dir / "splice_sites_enhanced.tsv"

        if not path.exists():
            raise FileNotFoundError(
                f"Splice sites not found at {path}. "
                f"Run: agentic-spliceai-prepare --annotation-source "
                f"{resources.annotation_source} --build {resources.build}"
            )

        return path

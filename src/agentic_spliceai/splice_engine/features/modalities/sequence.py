"""Sequence modality — contextual DNA sequence extraction from FASTA.

Extracts a fixed-length DNA window around each position using pyfaidx
for efficient random FASTA access. Adds ``sequence``, ``window_start``,
and ``window_end`` columns.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


@dataclass
class SequenceConfig(ModalityConfig):
    """Configuration for the sequence modality.

    Attributes
    ----------
    window_size : int
        Number of base pairs on each side of the position.
        Total sequence length = 2 * window_size + 1.
        Default 500 gives a 1001nt window (matching meta-spliceai).
    fasta_path : Path or None
        Path to reference FASTA file. If None, auto-resolved
        from the genomic registry based on base_model.
    base_model : str
        Base model name for auto-resolving paths.
    """

    window_size: int = 500
    fasta_path: Optional[Path] = None
    base_model: str = "openspliceai"


class SequenceModality(Modality):
    """Extract contextual DNA sequences from a reference FASTA.

    For each (chrom, position), extracts a window of
    ``2 * window_size + 1`` nucleotides centered on the position.
    Handles chromosome boundaries by clipping to valid ranges.
    """

    def __init__(self, config: SequenceConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: SequenceConfig = self.config  # type: ignore[assignment]
        self._fasta = None  # Lazy-loaded pyfaidx.Fasta

    @property
    def meta(self) -> ModalityMeta:
        return ModalityMeta(
            name="sequence",
            version="1.0",
            output_columns=("sequence", "window_start", "window_end"),
            required_inputs=frozenset({"chrom", "position"}),
            description=(
                f"Contextual DNA sequence extraction "
                f"(window={self._cfg.window_size}bp each side)."
            ),
        )

    @classmethod
    def default_config(cls) -> SequenceConfig:
        return SequenceConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)
        try:
            self._get_fasta()
        except Exception as e:
            errors.append(f"Cannot open FASTA file: {e}")
        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract DNA sequences for each position in the DataFrame.

        Processes genes in groups for I/O locality (positions within
        a gene are typically close together on the chromosome).
        """
        fasta = self._get_fasta()
        w = self._cfg.window_size

        # Extract sequences for all positions
        chroms = df["chrom"].to_list()
        positions = df["position"].to_list()

        sequences: list[str] = []
        window_starts: list[int] = []
        window_ends: list[int] = []

        for chrom, pos in zip(chroms, positions):
            pos = int(pos)  # ensure integer (may be float64 from TSV)
            fasta_chrom = self._resolve_fasta_chrom(fasta, chrom)
            chrom_len = len(fasta[fasta_chrom])

            # Clip to chromosome boundaries
            start = max(0, pos - w)
            end = min(chrom_len, pos + w)

            seq = str(fasta[fasta_chrom][start:end]).upper()

            # Pad if clipped at boundaries
            left_pad = (pos - w) - start  # 0 if no clip
            if left_pad < 0:
                seq = "N" * (-left_pad) + seq
                start = pos - w

            right_deficit = (pos + w) - end
            if right_deficit > 0:
                seq = seq + "N" * right_deficit
                end = pos + w

            sequences.append(seq)
            window_starts.append(start)
            window_ends.append(end)

        return df.with_columns(
            pl.Series("sequence", sequences),
            pl.Series("window_start", window_starts),
            pl.Series("window_end", window_ends),
        )

    def _get_fasta(self):
        """Get or lazily load the pyfaidx.Fasta handle."""
        if self._fasta is not None:
            return self._fasta

        import pyfaidx

        path = self._resolve_fasta_path()
        logger.info("Opening FASTA: %s", path)
        self._fasta = pyfaidx.Fasta(str(path))
        return self._fasta

    def _resolve_fasta_chrom(self, fasta, chrom: str) -> str:
        """Resolve chromosome name to match FASTA naming convention.

        Tries the given name, then with/without 'chr' prefix.
        """
        if chrom in fasta:
            return chrom
        alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        if alt in fasta:
            return alt
        raise KeyError(
            f"Chromosome '{chrom}' not found in FASTA. "
            f"Available: {list(fasta.keys())[:5]}..."
        )

    def _get_chrom_length(self, fasta, chrom: str) -> int:
        """Get chromosome length, trying with and without 'chr' prefix."""
        return len(fasta[self._resolve_fasta_chrom(fasta, chrom)])

    def _resolve_fasta_path(self) -> Path:
        """Resolve the path to the reference FASTA."""
        if self._cfg.fasta_path is not None:
            path = Path(self._cfg.fasta_path)
            if not path.exists():
                raise FileNotFoundError(f"FASTA file not found: {path}")
            return path

        # Auto-resolve from registry
        from agentic_spliceai.splice_engine.resources import get_model_resources

        resources = get_model_resources(self._cfg.base_model)
        fasta_path = resources.get_fasta_path()

        if fasta_path is None:
            raise FileNotFoundError(
                f"FASTA file not found for model '{self._cfg.base_model}'. "
                "Set fasta_path in SequenceConfig or configure the registry."
            )

        return Path(fasta_path)

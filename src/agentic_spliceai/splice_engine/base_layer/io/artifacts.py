"""Artifact management for chunked prediction workflows.

Handles structured output storage with mode-based overwrite policies:
- **production**: Immutable — refuses to overwrite existing artifacts
- **test**: Overwritable — allows re-running with fresh output

Directory layout::

    {output_dir}/
    ├── predictions_chunk_000.tsv   # Raw per-nucleotide scores (meta layer input)
    ├── predictions_chunk_001.tsv
    ├── positions_chunk_000.tsv     # Classified positions (TP/FP/FN)
    ├── positions_chunk_001.tsv
    ├── predictions.tsv             # Aggregated raw scores
    ├── positions.tsv               # Aggregated classified positions
    ├── manifest.tsv                # Gene processing manifest
    └── summary.json                # Workflow metadata and statistics
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages structured output for chunked prediction workflows.

    Parameters
    ----------
    output_dir : str or Path
        Root directory for artifacts.
    model_name : str
        Base model name (e.g., 'openspliceai', 'spliceai').
    genomic_build : str
        Genomic build (e.g., 'GRCh38', 'GRCh37').
    mode : str, default='test'
        Overwrite policy: 'production' (immutable) or 'test' (overwritable).

    Examples
    --------
    >>> am = ArtifactManager("output/run_001", "openspliceai", "GRCh38")
    >>> am.save_chunk(predictions_df, chunk_idx=0, artifact_type="predictions")
    >>> am.chunk_exists(chunk_idx=0, artifact_type="predictions")
    True
    """

    ARTIFACT_TYPES = ("predictions", "positions")

    def __init__(
        self,
        output_dir: Path | str,
        model_name: str,
        genomic_build: str,
        mode: str = "test",
        resume: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.genomic_build = genomic_build
        self.mode = mode
        self.resume = resume

        if self.mode not in ("production", "test"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'production' or 'test'.")

        self._ensure_output_dir()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def get_chunk_path(self, chunk_idx: int, artifact_type: str = "predictions") -> Path:
        """Get the file path for a specific chunk artifact.

        Parameters
        ----------
        chunk_idx : int
            Zero-based chunk index.
        artifact_type : str
            One of 'predictions' (raw scores) or 'positions' (classified).
        """
        self._validate_artifact_type(artifact_type)
        return self.output_dir / f"{artifact_type}_chunk_{chunk_idx:03d}.tsv"

    def get_aggregated_path(self, artifact_type: str = "predictions") -> Path:
        """Get the path for the aggregated (all-chunk) artifact."""
        self._validate_artifact_type(artifact_type)
        return self.output_dir / f"{artifact_type}.tsv"

    def get_manifest_path(self) -> Path:
        """Get the path for the gene manifest."""
        return self.output_dir / "manifest.tsv"

    def get_summary_path(self) -> Path:
        """Get the path for the workflow summary JSON."""
        return self.output_dir / "summary.json"

    # ------------------------------------------------------------------
    # Checkpoint queries
    # ------------------------------------------------------------------

    def chunk_exists(self, chunk_idx: int, artifact_type: str = "predictions") -> bool:
        """Check whether a chunk artifact already exists on disk."""
        return self.get_chunk_path(chunk_idx, artifact_type).exists()

    def count_existing_chunks(self, artifact_type: str = "predictions") -> int:
        """Count how many chunk files exist for the given artifact type."""
        pattern = f"{artifact_type}_chunk_*.tsv"
        return len(list(self.output_dir.glob(pattern)))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_chunk(
        self,
        df: pl.DataFrame,
        chunk_idx: int,
        artifact_type: str = "predictions",
    ) -> Path:
        """Save a chunk DataFrame to disk as TSV.

        Parameters
        ----------
        df : pl.DataFrame
            Chunk data to persist.
        chunk_idx : int
            Zero-based chunk index.
        artifact_type : str
            'predictions' or 'positions'.

        Returns
        -------
        Path
            Path where the chunk was saved.

        Raises
        ------
        FileExistsError
            If mode is 'production' and the chunk already exists.
        """
        path = self.get_chunk_path(chunk_idx, artifact_type)
        self._check_overwrite(path)
        self._atomic_write_tsv(df, path)
        logger.debug("Saved %s chunk %d (%d rows) → %s", artifact_type, chunk_idx, df.height, path)
        return path

    def load_chunk(self, chunk_idx: int, artifact_type: str = "predictions") -> pl.DataFrame:
        """Load a previously saved chunk from disk.

        Raises
        ------
        FileNotFoundError
            If the chunk file does not exist.
        """
        path = self.get_chunk_path(chunk_idx, artifact_type)
        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")
        return pl.read_csv(path, separator="\t")

    def save_aggregated(self, df: pl.DataFrame, artifact_type: str = "predictions") -> Path:
        """Save the final aggregated artifact (all chunks concatenated)."""
        path = self.get_aggregated_path(artifact_type)
        self._check_overwrite(path, is_aggregated=True)
        self._atomic_write_tsv(df, path)
        logger.info("Saved aggregated %s (%d rows) → %s", artifact_type, df.height, path)
        return path

    def save_manifest(self, manifest_df: pl.DataFrame) -> Path:
        """Save the gene processing manifest."""
        path = self.get_manifest_path()
        self._check_overwrite(path, is_aggregated=True)
        self._atomic_write_tsv(manifest_df, path)
        logger.info("Saved manifest (%d genes) → %s", manifest_df.height, path)
        return path

    def save_summary(self, summary: Dict[str, Any]) -> Path:
        """Save workflow summary as JSON."""
        path = self.get_summary_path()
        self._check_overwrite(path, is_aggregated=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Saved summary → %s", path)
        return path

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def load_all_chunks(self, artifact_type: str = "predictions") -> pl.DataFrame:
        """Load and concatenate all existing chunk files for an artifact type.

        Returns
        -------
        pl.DataFrame
            Concatenated DataFrame from all chunks, in chunk order.
        """
        pattern = f"{artifact_type}_chunk_*.tsv"
        chunk_files = sorted(self.output_dir.glob(pattern))
        if not chunk_files:
            return pl.DataFrame()

        frames = [pl.read_csv(f, separator="\t") for f in chunk_files]
        return pl.concat(frames)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atomic_write_tsv(self, df: pl.DataFrame, path: Path) -> None:
        """Write DataFrame to TSV atomically (temp file + rename).

        Prevents half-written files from being mistaken for valid
        checkpoints after a crash or standby interruption.
        """
        tmp_path = path.with_suffix(".tsv.tmp")
        df.write_csv(tmp_path, separator="\t")
        tmp_path.rename(path)

    def _ensure_output_dir(self) -> None:
        """Create the output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check_overwrite(self, path: Path, is_aggregated: bool = False) -> None:
        """Enforce overwrite policy.

        In production mode, raises FileExistsError if file already exists,
        unless ``resume=True`` and the file is an aggregated artifact
        (predictions.tsv, manifest.tsv, summary.json). These are re-generated
        from all chunks when resuming with new chromosomes.

        In test mode, overwrites silently.
        """
        if not path.exists() or self.mode == "test":
            return
        if self.resume and is_aggregated:
            logger.info("Resume mode: overwriting aggregated artifact %s", path.name)
            return
        raise FileExistsError(
            f"Production mode: refusing to overwrite existing artifact '{path}'. "
            "Use mode='test' to allow overwrites or delete the file manually."
        )

    def _validate_artifact_type(self, artifact_type: str) -> None:
        if artifact_type not in self.ARTIFACT_TYPES:
            raise ValueError(
                f"Invalid artifact_type '{artifact_type}'. "
                f"Must be one of: {self.ARTIFACT_TYPES}"
            )

    def __repr__(self) -> str:
        n_pred = self.count_existing_chunks("predictions")
        n_pos = self.count_existing_chunks("positions")
        return (
            f"ArtifactManager(dir='{self.output_dir}', mode='{self.mode}', "
            f"chunks: {n_pred} predictions, {n_pos} positions)"
        )

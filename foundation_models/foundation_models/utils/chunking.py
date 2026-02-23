"""
Sequence chunking utilities for Evo2 and other long-context genomic models.

Evo2 accepts up to 1 M bp in a single forward pass, but practical limits on
M1 Mac (~32 kb) or even GPU (~128 kb) mean long genes must be split.  This
module handles:

  1. Splitting a DNA sequence into overlapping windows.
  2. Reconstructing per-nucleotide embeddings from chunked results.
  3. Building gene-level windows from pre-extracted parquet files
     (``gene_sequence_*.parquet``) that already exist in the data directory.
  4. Deriving per-nucleotide exon/intron labels from ``splice_sites_enhanced.tsv``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Low-level chunk data structure
# ---------------------------------------------------------------------------

@dataclass
class SequenceChunk:
    """A single chunk of a longer DNA sequence.

    Attributes
    ----------
    sequence:
        The nucleotide substring for this chunk.
    global_start:
        0-based start position in the **original** (full-gene) sequence.
    global_end:
        Exclusive end position in the original sequence.
    chunk_idx:
        Index of this chunk in the ordered list of chunks for the gene.
    keep_start:
        Local offset (within *sequence*) of the first position that should be
        retained when stitching results back together.  Positions before this
        are overlap context from the previous chunk.
    keep_end:
        Local offset (exclusive) of the last position to retain.
    """
    sequence: str
    global_start: int
    global_end: int
    chunk_idx: int
    keep_start: int = 0
    keep_end: int = field(default=-1)  # -1 means "to the end"

    def __post_init__(self) -> None:
        if self.keep_end == -1:
            self.keep_end = len(self.sequence)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def keep_length(self) -> int:
        return self.keep_end - self.keep_start


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def chunk_sequence(
    sequence: str,
    chunk_size: int,
    overlap: int = 0,
) -> List[SequenceChunk]:
    """Split *sequence* into overlapping chunks of at most *chunk_size* bp.

    Parameters
    ----------
    sequence:
        Full DNA sequence string.
    chunk_size:
        Maximum length of each chunk.
    overlap:
        Number of bases of overlap between consecutive chunks.  Overlapping
        bases provide context to the model at chunk boundaries; they are
        trimmed when stitching embeddings back together.

    Returns
    -------
    List of :class:`SequenceChunk` in order.

    Notes
    -----
    Overlap is split symmetrically: the first ``overlap // 2`` bases of a
    chunk (except the very first) are context from the previous chunk and are
    discarded during stitching.  The last ``overlap // 2`` bases of a chunk
    (except the very last) are context for the next chunk and are also
    discarded.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    seq_len = len(sequence)
    if seq_len == 0:
        return []
    if seq_len <= chunk_size:
        return [SequenceChunk(sequence, 0, seq_len, 0, 0, seq_len)]

    step = chunk_size - overlap
    half_overlap = overlap // 2
    chunks: List[SequenceChunk] = []
    chunk_idx = 0

    start = 0
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        chunk_seq = sequence[start:end]

        # keep_start: skip overlapping prefix (except for first chunk)
        keep_start = half_overlap if (start > 0 and overlap > 0) else 0

        # keep_end: stop before overlapping suffix (except for last chunk)
        is_last = (end >= seq_len)
        keep_end = len(chunk_seq) if is_last else len(chunk_seq) - half_overlap

        chunks.append(SequenceChunk(
            sequence=chunk_seq,
            global_start=start,
            global_end=end,
            chunk_idx=chunk_idx,
            keep_start=keep_start,
            keep_end=keep_end,
        ))
        chunk_idx += 1

        if end >= seq_len:
            break
        start += step

    return chunks


def iter_chunks(
    sequence: str,
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[SequenceChunk]:
    """Lazy iterator version of :func:`chunk_sequence`."""
    yield from chunk_sequence(sequence, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Embedding stitching
# ---------------------------------------------------------------------------

def stitch_embeddings(
    chunks: List[SequenceChunk],
    chunk_embeddings: List[np.ndarray],
    seq_len: int,
    hidden_dim: int,
) -> np.ndarray:
    """Reassemble per-nucleotide embeddings from chunked outputs.

    Parameters
    ----------
    chunks:
        The :class:`SequenceChunk` objects returned by :func:`chunk_sequence`.
    chunk_embeddings:
        List of numpy arrays, one per chunk, shape ``[chunk_len, hidden_dim]``.
    seq_len:
        Expected length of the full reconstructed sequence.
    hidden_dim:
        Embedding dimension.

    Returns
    -------
    ``np.ndarray`` of shape ``[seq_len, hidden_dim]``.
    """
    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) != "
            f"number of embeddings ({len(chunk_embeddings)})"
        )

    result = np.zeros((seq_len, hidden_dim), dtype=np.float32)
    filled = np.zeros(seq_len, dtype=bool)

    for chunk, emb in zip(chunks, chunk_embeddings):
        # Slice the "keep" portion of this chunk's embeddings
        keep_emb = emb[chunk.keep_start : chunk.keep_end]  # [keep_len, H]
        global_s = chunk.global_start + chunk.keep_start
        global_e = global_s + len(keep_emb)

        if global_e > seq_len:
            keep_emb = keep_emb[:seq_len - global_s]
            global_e = seq_len

        result[global_s:global_e] = keep_emb
        filled[global_s:global_e] = True

    n_unfilled = (~filled).sum()
    if n_unfilled > 0:
        warnings.warn(
            f"{n_unfilled} positions have no embedding after stitching. "
            "They will remain zero-filled."
        )

    return result


# ---------------------------------------------------------------------------
# Gene-level helpers (interface to existing data infrastructure)
# ---------------------------------------------------------------------------

def load_gene_sequences_parquet(parquet_path: str) -> "pd.DataFrame":
    """Load a ``gene_sequence_*.parquet`` file.

    Columns expected: gene_id, gene_name, seqname, strand, start, end, sequence

    Parameters
    ----------
    parquet_path:
        Full path to the parquet file (e.g. ``data/ensembl/GRCh37/gene_sequence_1.parquet``).

    Returns
    -------
    ``pandas.DataFrame`` with one row per gene.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError("pyarrow is required: pip install pyarrow")

    df = pd.read_parquet(parquet_path)
    required_cols = {"gene_id", "gene_name", "seqname", "strand", "start", "end", "sequence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet file is missing expected columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    return df


def make_chunks_for_gene(
    gene_row: "pd.Series",
    chunk_size: int,
    overlap: int = 0,
) -> List[SequenceChunk]:
    """Return chunks for a single gene row from ``gene_sequence_*.parquet``.

    Parameters
    ----------
    gene_row:
        A single row from the gene-sequences DataFrame.
    chunk_size:
        Maximum chunk size in bp.
    overlap:
        Overlap between consecutive chunks.

    Returns
    -------
    List of :class:`SequenceChunk`.
    """
    sequence = gene_row["sequence"]
    if not isinstance(sequence, str):
        raise TypeError(
            f"Expected sequence to be str, got {type(sequence).__name__} "
            f"for gene {gene_row.get('gene_id', '?')}"
        )
    return chunk_sequence(sequence, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Exon label derivation from splice_sites_enhanced.tsv
# ---------------------------------------------------------------------------

def build_exon_labels(
    gene_id: str,
    gene_start: int,
    gene_sequence_length: int,
    splice_sites_df: "pd.DataFrame",
) -> np.ndarray:
    """Create a per-nucleotide binary exon/intron label vector.

    Uses annotated donor (end-of-exon) and acceptor (start-of-exon) positions
    from ``splice_sites_enhanced.tsv`` to mark exonic positions as 1 and
    intronic positions as 0.

    The label is relative to the gene sequence extracted in
    ``gene_sequence_*.parquet``, i.e. position 0 in the label corresponds to
    ``gene_start`` in genomic coordinates.

    Parameters
    ----------
    gene_id:
        Ensembl gene ID (used to filter *splice_sites_df*).
    gene_start:
        Genomic start coordinate of the gene (0-based, as stored in parquet).
    gene_sequence_length:
        Length of the gene sequence string.
    splice_sites_df:
        DataFrame loaded from ``splice_sites_enhanced.tsv``.
        Must have columns: gene_id, position, site_type, transcript_id,
        exon_number (or exon_rank), strand.

    Returns
    -------
    Binary ``np.ndarray`` of shape ``[gene_sequence_length]``, dtype uint8.
    1 = exon, 0 = intron/unknown.

    Notes
    -----
    Strategy:
    - For each transcript of the gene, sort exons by exon_rank.
    - Pair consecutive (acceptor[i], donor[i]) positions to define exon spans.
    - Mark all nucleotides within those spans as exonic.
    - The union across all transcripts is returned.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    labels = np.zeros(gene_sequence_length, dtype=np.uint8)

    gene_sites = splice_sites_df[splice_sites_df["gene_id"] == gene_id].copy()
    if gene_sites.empty:
        return labels

    rank_col = "exon_rank" if "exon_rank" in gene_sites.columns else "exon_number"

    for transcript_id, tx_sites in gene_sites.groupby("transcript_id"):
        tx_sites = tx_sites.sort_values(rank_col)

        donors = tx_sites[tx_sites["site_type"] == "donor"]
        acceptors = tx_sites[tx_sites["site_type"] == "acceptor"]

        # Build exon spans: each exon has one acceptor (start) and one donor (end).
        # For multi-exon transcripts, sorted by exon_rank:
        #   exon 1: [tx_start ... donor_1]
        #   exon 2: [acceptor_1 ... donor_2]
        #   ...
        #   exon n: [acceptor_{n-1} ... tx_end]
        #
        # We reconstruct this from the sorted donor/acceptor pairs.
        donor_positions = sorted(donors["position"].tolist())
        acceptor_positions = sorted(acceptors["position"].tolist())

        strand = tx_sites["strand"].iloc[0]

        # Build spans as (start_genomic, end_genomic) inclusive pairs
        spans: List[Tuple[int, int]] = []

        if strand == "+":
            # Exons run left-to-right
            # Interleave: exon_start ... donor, acceptor ... exon_end
            # First exon ends at donor_positions[0]
            # For simplicity, treat each consecutive acceptor-donor pair as an exon body
            for acc, don in zip(acceptor_positions, donor_positions):
                if acc <= don:
                    spans.append((acc, don))
        else:
            # Minus strand: donors < acceptors in genomic coords
            # (donor is actually the 3' end of the exon on genomic coords)
            for don, acc in zip(donor_positions, acceptor_positions):
                if don <= acc:
                    spans.append((don, acc))

        for gstart, gend in spans:
            # Convert to relative coordinates within the gene sequence
            rel_start = gstart - gene_start
            rel_end = gend - gene_start + 1  # inclusive → exclusive

            # Clip to gene sequence bounds
            rel_start = max(0, rel_start)
            rel_end = min(gene_sequence_length, rel_end)

            if rel_start < rel_end:
                labels[rel_start:rel_end] = 1

    return labels


# ---------------------------------------------------------------------------
# Window iterator for training data generation
# ---------------------------------------------------------------------------

@dataclass
class LabeledWindow:
    """A sequence window with per-nucleotide exon labels.

    Used as training examples for the exon classifier.

    Attributes
    ----------
    sequence:
        DNA subsequence of length *window_size*.
    labels:
        Binary array of shape ``[window_size]`` (1 = exon, 0 = intron).
    gene_id:
        Source gene identifier.
    chrom:
        Chromosome (seqname).
    global_start:
        0-based genomic start of this window.
    """
    sequence: str
    labels: np.ndarray
    gene_id: str
    chrom: str
    global_start: int


def generate_labeled_windows(
    gene_sequence: str,
    exon_labels: np.ndarray,
    gene_id: str,
    chrom: str,
    gene_genomic_start: int,
    window_size: int,
    step_size: Optional[int] = None,
    min_exon_fraction: float = 0.0,
) -> Iterator[LabeledWindow]:
    """Yield fixed-size labeled windows from a gene sequence.

    Parameters
    ----------
    gene_sequence:
        Full gene nucleotide sequence.
    exon_labels:
        Binary per-nucleotide labels from :func:`build_exon_labels`.
    gene_id:
        Ensembl gene ID string.
    chrom:
        Chromosome name.
    gene_genomic_start:
        Genomic start of the gene (for coordinate reporting).
    window_size:
        Size of each window in bp.
    step_size:
        Stride between windows.  Defaults to *window_size* (non-overlapping).
    min_exon_fraction:
        Skip windows where the fraction of exonic nucleotides is below this
        threshold.  Useful to filter out purely intronic windows when
        the training set is already unbalanced.

    Yields
    ------
    :class:`LabeledWindow` instances.
    """
    if step_size is None:
        step_size = window_size

    seq_len = len(gene_sequence)

    for start in range(0, seq_len - window_size + 1, step_size):
        end = start + window_size
        win_seq = gene_sequence[start:end]
        win_labels = exon_labels[start:end]

        if min_exon_fraction > 0:
            exon_frac = win_labels.mean()
            if exon_frac < min_exon_fraction:
                continue

        yield LabeledWindow(
            sequence=win_seq,
            labels=win_labels,
            gene_id=gene_id,
            chrom=chrom,
            global_start=gene_genomic_start + start,
        )


# ---------------------------------------------------------------------------
# Utility: estimate number of chunks / windows for a dataset
# ---------------------------------------------------------------------------

def estimate_chunk_count(
    total_sequence_length: int,
    chunk_size: int,
    overlap: int = 0,
) -> int:
    """Estimate the number of chunks that will be produced."""
    if total_sequence_length <= chunk_size:
        return 1
    step = chunk_size - overlap
    return max(1, int(np.ceil(total_sequence_length / step)))


def estimate_window_count(
    total_sequence_length: int,
    window_size: int,
    step_size: Optional[int] = None,
) -> int:
    """Estimate the number of labeled windows for a given sequence length."""
    if step_size is None:
        step_size = window_size
    if total_sequence_length < window_size:
        return 0
    return max(0, (total_sequence_length - window_size) // step_size + 1)

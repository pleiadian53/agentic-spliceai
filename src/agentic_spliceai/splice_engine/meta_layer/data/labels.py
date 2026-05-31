"""Per-position label builders for the meta-layer sequence models.

`build_splice_labels` was copied verbatim from
``foundation_models/foundation_models/utils/chunking.py`` so that `src/` no
longer has to import from the experimental ``foundation_models/`` sub-project
(see memory ``feedback_no_foundation_models_dep_in_src``). The new
``build_m3_labels`` is added alongside it for the M3 novel-site recognizer.

Label conventions
-----------------
``build_splice_labels`` uses the original *chunking* convention:
    0 = no splice site, 1 = acceptor, 2 = donor.
The training pipeline then remaps this to the *meta-layer* convention via
``_CHUNKING_TO_META`` in ``sequence_level_dataset.py``:
    0 = donor, 1 = acceptor, 2 = neither.

``build_m3_labels`` emits the *meta-layer* convention DIRECTLY (same 3-class
shape as M1-S/M2-S) plus a ``255 = ignore`` sentinel for annotated positions,
which the recognizer + post-filter framing in M3 wants masked out of the loss
(``CrossEntropyLoss(ignore_index=255)``). The M3 cache-build path skips the
chunking→meta remap.

The M3 label set is keyed by GENOMIC COORDINATE only — ``(chrom, position,
splice_type)``, with **no gene_id** — so ``build_m3_labels`` assigns labels to a
gene window by interval overlap (``position - gene_start``), not by gene id.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import warnings

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Meta-layer 3-class convention (donor=0, acceptor=1, neither=2). M3 adds:
IGNORE_INDEX: int = 255  # uint8 sentinel; loss passes ignore_index=255 to CE


def build_splice_labels(
    gene_id: str,
    gene_start: int,
    gene_sequence_length: int,
    splice_sites_df: "pd.DataFrame",
) -> np.ndarray:
    """Create per-nucleotide 3-class splice site labels (chunking convention).

    Marks exact splice junction positions:
      - 0 = no splice site (vast majority)
      - 1 = acceptor (first nucleotide of exon, 3' splice site)
      - 2 = donor (last nucleotide of exon, 5' splice site)

    Coordinates follow the same convention as ``build_exon_labels`` — position
    0 in the returned array corresponds to ``gene_start`` in genomic coords.

    Parameters
    ----------
    gene_id:
        Ensembl gene ID (used to filter *splice_sites_df*).
    gene_start:
        Genomic start coordinate of the gene (0-based, as stored in parquet).
    gene_sequence_length:
        Length of the gene sequence string.
    splice_sites_df:
        DataFrame loaded from ``splice_sites_enhanced.tsv`` — needs columns
        ``gene_id, position, splice_type``.

    Returns
    -------
    ``np.ndarray`` of shape ``[gene_sequence_length]``, dtype uint8.
    Values: 0=no_splice, 1=acceptor, 2=donor.

    Notes
    -----
    - Union across transcripts: if a position is a donor in any transcript,
      it is marked as donor.
    - On per-position conflict (donor in one transcript, acceptor in another),
      the first-encountered label wins and a warning is emitted.
    """
    labels = np.zeros(gene_sequence_length, dtype=np.uint8)

    gene_sites = splice_sites_df[splice_sites_df["gene_id"] == gene_id]
    if gene_sites.empty:
        return labels

    splice_type_map = {"acceptor": 1, "donor": 2}
    n_conflicts = 0

    for _, row in gene_sites.iterrows():
        splice_type = row["splice_type"]
        label_val = splice_type_map.get(splice_type)
        if label_val is None:
            continue

        rel_pos = row["position"] - gene_start

        if rel_pos < 0 or rel_pos >= gene_sequence_length:
            continue

        if labels[rel_pos] != 0 and labels[rel_pos] != label_val:
            n_conflicts += 1
            continue

        labels[rel_pos] = label_val

    if n_conflicts > 0:
        warnings.warn(
            f"Gene {gene_id}: {n_conflicts} positions had conflicting "
            f"splice_type labels (kept first-encountered)",
            stacklevel=2,
        )

    return labels


def build_m3_labels(
    gene_start: int,
    gene_sequence_length: int,
    positives_df: "pd.DataFrame",
    annotation_mask_df: "pd.DataFrame",
    confirmed_weight: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-position 3-class M3 labels + per-position weights (meta convention).

    Output convention (training-ready, identical shape to M1-S/M2-S; no remap):
      - 0 = donor (novel donor at an M3 positive position)
      - 1 = acceptor (novel acceptor at an M3 positive position)
      - 2 = neither (default for EVERY non-labeled position in the window)
      - 255 = ignore — annotated splice site; loss uses ``ignore_index=255``.

    The "recognizer + post-filter" framing for M3: novel sites are positives,
    annotated sites are masked out of the loss (a novel and an annotated donor
    are sequence-identical → contradictory labels otherwise), and "novelty" is
    applied at inference as exact set subtraction, not learned. Every other
    position in the gene window is automatically class 2 (neither) — there is
    no separate negative-sampling step.

    Coordinate assignment
    ---------------------
    The M3 label set has **no gene_id** — it is keyed by genomic coordinate.
    Positions are mapped into the window by ``rel = position - gene_start``
    (same origin as ``build_splice_labels`` and the gene's FASTA slice).
    *positives_df* and *annotation_mask_df* MUST already be subset to this
    gene's chromosome by the caller (bare chrom, e.g. ``"1"`` not ``"chr1"``);
    this function applies only the position-range filter. Strand is not used —
    matching ``build_splice_labels`` (M1/M2), the model trains on forward-strand
    sequence and labels sit at genomic positions.

    Parameters
    ----------
    gene_start:
        Genomic start coordinate of the gene (0-based), the rel origin.
    gene_sequence_length:
        Length of the gene window.
    positives_df:
        Novel positives for THIS chromosome. Required columns:
        ``position, splice_type`` (donor/acceptor). Optional
        ``longread_confirmed`` (bool) — True → ``confirmed_weight`` (else 1.0).
    annotation_mask_df:
        Annotated splice sites for THIS chromosome (the loss-ignore mask).
        Required column: ``position`` (``splice_type`` ignored — any annotated
        position is masked regardless of donor/acceptor identity).
    confirmed_weight:
        Per-position weight applied to long-read-confirmed positives.

    Returns
    -------
    labels:
        ``[gene_sequence_length]`` uint8, meta convention with the IGNORE
        sentinel.
    weights:
        ``[gene_sequence_length]`` float32. ``confirmed_weight`` at confirmed
        positive positions, ``1.0`` elsewhere.
    """
    labels = np.full(gene_sequence_length, 2, dtype=np.uint8)  # default = neither
    weights = np.ones(gene_sequence_length, dtype=np.float32)

    gene_end = gene_start + gene_sequence_length  # 0-based, exclusive

    # 1. Annotated positions in this gene's window → IGNORE.
    if annotation_mask_df is not None and len(annotation_mask_df) > 0:
        ann_pos = annotation_mask_df["position"].to_numpy(dtype=np.int64)
        sel = (ann_pos >= gene_start) & (ann_pos < gene_end)
        if sel.any():
            labels[ann_pos[sel] - gene_start] = IGNORE_INDEX

    # 2. Novel positives → donor(0) / acceptor(1); confirmed → up-weighted.
    #    Positives override IGNORE (defensive — positives are annotation-clean
    #    by construction in B1, so this should never trigger in practice).
    if positives_df is None or len(positives_df) == 0:
        return labels, weights

    pos = positives_df["position"].to_numpy(dtype=np.int64)
    in_win = (pos >= gene_start) & (pos < gene_end)
    if not in_win.any():
        return labels, weights

    sub = positives_df[in_win]
    rel_all = sub["position"].to_numpy(dtype=np.int64) - gene_start
    stype = sub["splice_type"].to_numpy()
    has_conf = "longread_confirmed" in sub.columns
    conf_all = (
        sub["longread_confirmed"].to_numpy(dtype=bool) if has_conf else None
    )

    for name, lab in (("donor", 0), ("acceptor", 1)):
        m = stype == name
        if not m.any():
            continue
        rel = rel_all[m]
        labels[rel] = lab
        if has_conf:
            conf = conf_all[m]
            if conf.any():
                weights[rel[conf]] = float(confirmed_weight)

    return labels, weights

"""HGVS coding-notation to genomic-coordinate resolution.

Resolves HGVS strings of the form
``NM_XXXXX.Y:c.NNN[+-]M[ACGT]>[ACGT]`` to exact genomic positions on a
reference assembly. Supports the SNV subset that covers ~100% of variant
catalogs we ingest (MutSpliceDB, SpliceVault):

- Pure exonic substitutions: ``c.236G>A``
- Intronic splice-region SNVs: ``c.236+1G>A`` (donor +1), ``c.2989-3C>G``
  (acceptor -3)

The resolver uses a ``gffutils`` ``FeatureDB`` built from a RefSeq or MANE
GFF/GTF for transcript-to-genomic mapping. Strand is read from the GTF —
caller-supplied strand fields (which are often unreliable in derived
catalogs) are ignored.

For minus-strand transcripts, HGVS alleles are in **transcript orientation**
(antisense to the genome). The resolver reverse-complements them so the
returned ``ref`` always matches FASTA at ``position``.

Example
-------
::

    resolver = HgvsResolver("data/mane/GRCh38/annotations.db")
    v = resolver.resolve("NM_005957.5:c.236+1G>A")
    # GenomicVariant(chrom='chr1', position=..., ref='C', alt='T',
    #                strand='-', transcript_id='NM_005957.5', ...)

See Also
--------
- ``splice_engine.meta_layer.data.mutsplicedb_loader`` — primary consumer.
- ``examples/data_preparation/variants/01_resolve_hgvs_to_genomic.py``
  — end-to-end example over the full MutSpliceDB TSV.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import gffutils

logger = logging.getLogger(__name__)

# Matches: NM_005957.5:c.236+1G>A, NM_005957.5:c.236G>A, NM_005957.5:c.2989-3C>G
#   tx     : the transcript accession (e.g. NM_005957.5)
#   pos    : the CDS-relative coding position (1-based)
#   sign   : '+' or '-' if intronic offset present, else None
#   offset : intron offset in bp from the exon boundary, else None
#   ref    : single-base HGVS ref allele (transcript orientation)
#   alt    : single-base HGVS alt allele (transcript orientation)
_HGVS_FULL = re.compile(
    r"^(?P<tx>[A-Za-z]+_\d+\.\d+):c\."
    r"(?P<pos>\d+)"
    r"(?:(?P<sign>[+-])(?P<offset>\d+))?"
    r"(?P<ref>[ACGT])>(?P<alt>[ACGT])$"
)
_RC = str.maketrans("ACGT", "TGCA")


def _reverse_complement(s: str) -> str:
    return s.translate(_RC)[::-1]


@dataclass(frozen=True)
class GenomicVariant:
    """A variant resolved to genomic-strand coordinates.

    Attributes
    ----------
    chrom : str
        Sequence id as it appears in the GTF (typically chr-prefixed,
        e.g. ``chr1``).
    position : int
        1-based genomic position of the variant.
    ref, alt : str
        Genomic-strand alleles. For minus-strand transcripts these are
        the reverse-complements of the HGVS-string alleles.
    strand : str
        Transcript strand ('+' or '-'), as read from the GTF.
    transcript_id : str
        Transcript accession parsed from the HGVS string.
    hgvs : str
        The original HGVS string for traceability.
    """

    chrom: str
    position: int
    ref: str
    alt: str
    strand: str
    transcript_id: str
    hgvs: str


class HgvsResolutionError(ValueError):
    """Raised when an HGVS string cannot be parsed or resolved."""


class HgvsResolver:
    """Resolve HGVS coding-notation strings to genomic coordinates.

    Parameters
    ----------
    db_path : Path or str
        Path to a ``gffutils.FeatureDB`` built from a RefSeq or MANE
        GFF/GTF. Transcript records are expected to have feature type
        ``transcript`` and may be keyed either bare (``NM_005957.5``)
        or with the NCBI GFF ``rna-`` prefix (``rna-NM_005957.5``).
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._db: Optional[gffutils.FeatureDB] = None

    @property
    def db(self) -> gffutils.FeatureDB:
        if self._db is None:
            self._db = gffutils.FeatureDB(str(self.db_path))
        return self._db

    def resolve(self, hgvs: str) -> GenomicVariant:
        """Resolve a full HGVS string to a genomic variant.

        Parameters
        ----------
        hgvs : str
            HGVS coding-notation string, e.g. ``"NM_005957.5:c.236+1G>A"``.

        Returns
        -------
        GenomicVariant
            Variant in genomic orientation.

        Raises
        ------
        HgvsResolutionError
            If the HGVS string fails parsing, the transcript is missing
            from the database, or the coding position lies beyond the
            transcript's CDS length.
        """
        m = _HGVS_FULL.match(hgvs.strip())
        if not m:
            raise HgvsResolutionError(f"Unparseable HGVS: {hgvs!r}")

        tx_id = m.group("tx")
        coding_pos = int(m.group("pos"))
        sign = m.group("sign")
        offset = int(m.group("offset")) if m.group("offset") else 0
        hgvs_ref = m.group("ref")
        hgvs_alt = m.group("alt")

        tx = self._lookup_transcript(tx_id)
        if tx is None:
            raise HgvsResolutionError(f"Transcript not in GTF: {tx_id}")

        # Walk CDS in transcript orientation to find the exonic anchor.
        exonic_genomic = self._coding_to_genomic(tx, coding_pos)

        # Apply intronic offset in transcript orientation.
        # +N (donor side): N bases past the 3' end of the upstream exon
        # -N (acceptor side): N bases before the 5' end of the downstream exon
        if sign == "+":
            genomic_pos = exonic_genomic + offset if tx.strand == "+" else exonic_genomic - offset
        elif sign == "-":
            genomic_pos = exonic_genomic - offset if tx.strand == "+" else exonic_genomic + offset
        else:
            genomic_pos = exonic_genomic

        # Translate alleles to genomic orientation (RC for minus-strand transcripts).
        if tx.strand == "-":
            g_ref, g_alt = _reverse_complement(hgvs_ref), _reverse_complement(hgvs_alt)
        else:
            g_ref, g_alt = hgvs_ref, hgvs_alt

        return GenomicVariant(
            chrom=tx.seqid,
            position=genomic_pos,
            ref=g_ref,
            alt=g_alt,
            strand=tx.strand,
            transcript_id=tx_id,
            hgvs=hgvs,
        )

    def resolve_or_none(self, hgvs: str) -> Optional[GenomicVariant]:
        """Same as :meth:`resolve` but returns ``None`` on failure (no raise).

        Useful for bulk processing where per-row failures should be tallied,
        not propagated.
        """
        try:
            return self.resolve(hgvs)
        except HgvsResolutionError as e:
            logger.debug("HGVS resolution failed: %s", e)
            return None

    @lru_cache(maxsize=8192)
    def _lookup_transcript(self, tx_id: str):
        """Find a transcript by accession.

        Resolution order:
        1. Direct primary-key lookup (``tx_id`` and ``rna-{tx_id}``).
        2. Version-agnostic prefix lookup: if the DB has any transcript
           whose id starts with ``rna-{bare_id}.``, return the first one
           and log a warning. This handles the common case where a
           variant catalog cites an older RefSeq version (e.g.
           ``NM_006015.4``) than the one MANE/RefSeq has now
           (``NM_006015.6``). CDS structure is usually stable across
           minor versions, so the genomic coord is typically still
           correct — the FASTA-ref-match check in the example script
           catches the rare cases where it isn't.
        3. Attribute LIKE search as a final fallback (slower).

        Returns the gffutils Feature, or ``None`` if not found.
        """
        for candidate in (tx_id, f"rna-{tx_id}"):
            try:
                feat = self.db[candidate]
                if feat.featuretype == "transcript":
                    return feat
            except gffutils.FeatureNotFoundError:
                continue

        # Version-agnostic id-prefix lookup.
        bare_id = tx_id.rsplit(".", 1)[0] if "." in tx_id else tx_id
        if bare_id != tx_id:
            for row in self.db.execute(
                f"SELECT id FROM features WHERE id LIKE 'rna-{bare_id}.%' "
                f"AND featuretype='transcript' LIMIT 1"
            ):
                resolved_id = row["id"]
                logger.warning(
                    "Version-fallback lookup: %s not in DB, using %s "
                    "(CDS structure assumed stable across versions)",
                    tx_id, resolved_id,
                )
                return self.db[resolved_id]

        # Final fallback: full-attributes search.
        for row in self.db.execute(
            "SELECT id FROM features WHERE featuretype='transcript' "
            f"AND attributes LIKE '%\"{tx_id}\"%' LIMIT 1"
        ):
            return self.db[row["id"]]
        return None

    def _coding_to_genomic(self, tx, coding_pos: int) -> int:
        """Translate a CDS-relative coding position to a genomic 1-based position.

        Walks the transcript's CDS records in transcript orientation,
        accumulating CDS length until ``coding_pos`` is reached. Returns
        the genomic coordinate of that exonic base — intronic offsets are
        applied by the caller.
        """
        cds_segments = list(self.db.children(tx, featuretype="CDS", order_by="start"))
        if not cds_segments:
            raise HgvsResolutionError(f"No CDS records for transcript {tx.id}")

        if tx.strand == "-":
            cds_segments.reverse()

        remaining = coding_pos
        for seg in cds_segments:
            seg_len = seg.end - seg.start + 1  # 1-based inclusive
            if remaining <= seg_len:
                if tx.strand == "+":
                    return seg.start + remaining - 1
                return seg.end - remaining + 1
            remaining -= seg_len

        raise HgvsResolutionError(
            f"Coding position {coding_pos} exceeds CDS length for transcript {tx.id}"
        )

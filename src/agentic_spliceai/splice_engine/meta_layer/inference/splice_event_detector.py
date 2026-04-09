"""Splice consequence prediction from variant delta scores.

Given a ``DeltaResult`` from :class:`VariantRunner` and gene structure
annotations, predicts the biological consequence of a variant on
splicing: exon skipping, intron retention, cryptic exon inclusion,
partial exon truncation, or donor/acceptor shift.

Also performs reading frame analysis when CDS annotations are available.

Usage::

    detector = SpliceEventDetector(gtf_path="data/mane/GRCh38/...")
    consequence = detector.analyze(delta_result, gene="MYBPC3")
    print(consequence.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExonInfo:
    """A single exon in a transcript."""

    exon_number: int
    start: int  # genomic start (0-based or 1-based, matching GTF)
    end: int  # genomic end
    strand: str
    transcript_id: str

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def donor_pos(self) -> int:
        """Genomic position of the donor site (3' end of exon)."""
        return self.end if self.strand == "+" else self.start

    @property
    def acceptor_pos(self) -> int:
        """Genomic position of the acceptor site (5' start of exon)."""
        return self.start if self.strand == "+" else self.end


@dataclass
class CDSRegion:
    """A CDS region within a transcript (coding portion of an exon)."""

    start: int
    end: int
    frame: int  # reading frame phase: 0, 1, or 2
    strand: str


@dataclass
class GeneStructure:
    """Complete gene structure for consequence analysis."""

    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    exons: List[ExonInfo]  # sorted by genomic position
    cds_regions: List[CDSRegion]  # sorted by genomic position
    transcript_id: str

    @property
    def n_exons(self) -> int:
        return len(self.exons)

    @property
    def has_cds(self) -> bool:
        return len(self.cds_regions) > 0

    @property
    def coding_length(self) -> int:
        """Total CDS length in bp."""
        return sum(r.end - r.start for r in self.cds_regions)

    def donor_positions(self) -> List[int]:
        """Genomic positions of all annotated donor sites."""
        if len(self.exons) < 2:
            return []
        # Donors at 3' end of all exons except the last
        sorted_exons = sorted(self.exons, key=lambda e: e.start)
        return [e.donor_pos for e in sorted_exons[:-1]]

    def acceptor_positions(self) -> List[int]:
        """Genomic positions of all annotated acceptor sites."""
        if len(self.exons) < 2:
            return []
        # Acceptors at 5' start of all exons except the first
        sorted_exons = sorted(self.exons, key=lambda e: e.start)
        return [e.acceptor_pos for e in sorted_exons[1:]]

    def find_nearest_exon(self, position: int) -> Optional[ExonInfo]:
        """Find the exon whose boundary is nearest to the given position."""
        best = None
        best_dist = float("inf")
        for exon in self.exons:
            for boundary in (exon.start, exon.end):
                d = abs(position - boundary)
                if d < best_dist:
                    best_dist = d
                    best = exon
        return best

    def is_position_in_cds(self, position: int) -> bool:
        """Check if a genomic position falls within a CDS region."""
        return any(r.start <= position <= r.end for r in self.cds_regions)


@dataclass
class JunctionChange:
    """A predicted change in splice junction usage."""

    junction_type: str  # "canonical_loss", "cryptic_gain", "shift"
    donor_pos: Optional[int]  # genomic position of the donor
    acceptor_pos: Optional[int]  # genomic position of the acceptor
    delta_score: float  # strength of evidence
    ref_junction: bool  # exists in reference annotation
    alt_junction: bool  # predicted in alternate allele
    intron_size: Optional[int]  # distance between donor and acceptor

    def __str__(self) -> str:
        kind = "LOSS" if self.junction_type == "canonical_loss" else "GAIN"
        pos = self.donor_pos or self.acceptor_pos
        return f"{kind}: {self.junction_type} at {pos} (Δ={self.delta_score:+.3f})"


@dataclass
class SpliceConsequence:
    """Full consequence prediction for a variant."""

    variant: str  # e.g. "chr11:47332565 C>A"
    gene: str
    consequence_type: str  # primary classification
    junction_changes: List[JunctionChange] = field(default_factory=list)
    affected_exons: List[int] = field(default_factory=list)  # exon numbers
    frame_preserved: Optional[bool] = None
    protein_impact: Optional[str] = None
    confidence: str = "LOW"  # LOW / MODERATE / HIGH
    details: List[str] = field(default_factory=list)  # detailed event descriptions

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        frame = ""
        if self.frame_preserved is True:
            frame = " (in-frame)"
        elif self.frame_preserved is False:
            frame = " (frameshift)"

        exon_str = ""
        if self.affected_exons:
            exon_str = f" affecting exon(s) {', '.join(str(e) for e in self.affected_exons)}"

        return (
            f"{self.consequence_type}{exon_str}{frame} "
            f"[{self.confidence} confidence]"
        )

    def report(self) -> str:
        """Multi-line detailed report."""
        lines = [
            f"Variant: {self.variant} ({self.gene})",
            f"Consequence: {self.consequence_type}",
            f"Confidence: {self.confidence}",
        ]
        if self.affected_exons:
            lines.append(f"Affected exons: {self.affected_exons}")
        if self.frame_preserved is not None:
            lines.append(
                f"Reading frame: {'preserved' if self.frame_preserved else 'disrupted (frameshift)'}"
            )
        if self.protein_impact:
            lines.append(f"Protein impact: {self.protein_impact}")
        if self.junction_changes:
            lines.append(f"\nJunction changes ({len(self.junction_changes)}):")
            for jc in self.junction_changes:
                lines.append(f"  {jc}")
        if self.details:
            lines.append(f"\nDetails:")
            for d in self.details:
                lines.append(f"  {d}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consequence type constants
# ---------------------------------------------------------------------------

EXON_SKIPPING = "exon_skipping"
INTRON_RETENTION = "intron_retention"
CRYPTIC_EXON = "cryptic_exon_inclusion"
DONOR_SHIFT = "donor_shift"
ACCEPTOR_SHIFT = "acceptor_shift"
DONOR_DESTRUCTION = "donor_destruction"
ACCEPTOR_DESTRUCTION = "acceptor_destruction"
COMPLEX = "complex_splicing_change"
NO_EFFECT = "no_significant_effect"

# Confidence thresholds (based on SpliceAI convention)
_HIGH_THRESHOLD = 0.5
_MODERATE_THRESHOLD = 0.2
_LOW_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# SpliceEventDetector
# ---------------------------------------------------------------------------


class SpliceEventDetector:
    """Predict splice consequences from variant delta scores.

    Parameters
    ----------
    gtf_path : Path or str
        Path to GTF annotation file for gene structure.
    build : str
        Genome build (default ``"GRCh38"``).
    max_pairing_distance : int
        Maximum distance (bp) between a loss and gain event to
        consider them paired (default 500).
    """

    def __init__(
        self,
        gtf_path: str | Path,
        build: str = "GRCh38",
        max_pairing_distance: int = 500,
    ) -> None:
        self.gtf_path = str(gtf_path)
        self.build = build
        self.max_pairing_distance = max_pairing_distance

        # Lazy-loaded caches
        self._exon_cache: Optional[pl.DataFrame] = None
        self._cds_cache: Optional[pl.DataFrame] = None
        self._gene_structure_cache: Dict[str, GeneStructure] = {}

    def _load_exons(self) -> pl.DataFrame:
        """Lazy-load exon annotations."""
        if self._exon_cache is None:
            from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
                extract_exon_annotations,
            )
            self._exon_cache = extract_exon_annotations(
                self.gtf_path, verbosity=0,
            )
        return self._exon_cache

    def _load_cds(self) -> pl.DataFrame:
        """Lazy-load CDS annotations."""
        if self._cds_cache is None:
            from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
                extract_cds_annotations,
            )
            self._cds_cache = extract_cds_annotations(
                self.gtf_path, verbosity=0,
            )
        return self._cds_cache

    def get_gene_structure(
        self, gene_name: str, transcript_id: Optional[str] = None,
    ) -> Optional[GeneStructure]:
        """Load gene structure for a gene, selecting canonical transcript.

        Parameters
        ----------
        gene_name : str
            Gene symbol (e.g. ``"MYBPC3"``).
        transcript_id : str, optional
            Force a specific transcript.  If None, selects the longest
            protein-coding transcript.

        Returns
        -------
        GeneStructure or None
            Gene structure with exon and CDS info, or None if not found.
        """
        cache_key = f"{gene_name}:{transcript_id or 'auto'}"
        if cache_key in self._gene_structure_cache:
            return self._gene_structure_cache[cache_key]

        exon_df = self._load_exons()

        # Find the gene
        gene_exons = exon_df.filter(pl.col("gene_name") == gene_name)
        if len(gene_exons) == 0:
            # Try gene_id
            gene_exons = exon_df.filter(pl.col("gene_id") == gene_name)
        if len(gene_exons) == 0:
            logger.warning("Gene %s not found in GTF", gene_name)
            return None

        # Select transcript
        if transcript_id:
            tx_exons = gene_exons.filter(pl.col("transcript_id") == transcript_id)
            if len(tx_exons) == 0:
                logger.warning("Transcript %s not found for %s", transcript_id, gene_name)
                return None
        else:
            # Select longest protein-coding transcript
            tx_lengths = (
                gene_exons.filter(pl.col("transcript_biotype") == "protein_coding")
                .group_by("transcript_id")
                .agg((pl.col("end") - pl.col("start")).sum().alias("total_length"))
                .sort("total_length", descending=True)
            )
            if len(tx_lengths) == 0:
                # Fall back to any transcript
                tx_lengths = (
                    gene_exons.group_by("transcript_id")
                    .agg((pl.col("end") - pl.col("start")).sum().alias("total_length"))
                    .sort("total_length", descending=True)
                )
            if len(tx_lengths) == 0:
                return None

            transcript_id = tx_lengths[0, "transcript_id"]
            tx_exons = gene_exons.filter(pl.col("transcript_id") == transcript_id)

        # Build exon list
        strand = tx_exons[0, "strand"]
        chrom = tx_exons[0, "chrom"]
        gene_id = tx_exons[0, "gene_id"]

        exons = []
        for row in tx_exons.sort("start").iter_rows(named=True):
            exons.append(
                ExonInfo(
                    exon_number=row.get("exon_number", 0) or len(exons) + 1,
                    start=row["start"],
                    end=row["end"],
                    strand=strand,
                    transcript_id=transcript_id,
                )
            )

        # Re-number exons if needed (1-based, in transcript order)
        if strand == "-":
            for i, exon in enumerate(reversed(exons)):
                exon.exon_number = i + 1
        else:
            for i, exon in enumerate(exons):
                exon.exon_number = i + 1

        # Load CDS regions for this transcript
        cds_regions = []
        try:
            cds_df = self._load_cds()
            tx_cds = cds_df.filter(pl.col("transcript_id") == transcript_id)
            for row in tx_cds.sort("start").iter_rows(named=True):
                cds_regions.append(
                    CDSRegion(
                        start=row["start"],
                        end=row["end"],
                        frame=row["frame"],
                        strand=strand,
                    )
                )
        except Exception as e:
            logger.debug("CDS loading failed for %s: %s", gene_name, e)

        structure = GeneStructure(
            gene_id=gene_id,
            gene_name=gene_name,
            chrom=chrom,
            strand=strand,
            exons=exons,
            cds_regions=cds_regions,
            transcript_id=transcript_id,
        )
        self._gene_structure_cache[cache_key] = structure
        return structure

    def analyze(
        self,
        delta_result: "DeltaResult",
        gene: Optional[str] = None,
        gene_structure: Optional[GeneStructure] = None,
    ) -> SpliceConsequence:
        """Analyze variant delta scores and predict splice consequence.

        Parameters
        ----------
        delta_result : DeltaResult
            Output from :meth:`VariantRunner.run`.
        gene : str, optional
            Gene name (used to load structure if ``gene_structure`` is None).
        gene_structure : GeneStructure, optional
            Pre-loaded gene structure.  If None, loaded from GTF using ``gene``.

        Returns
        -------
        SpliceConsequence
            Predicted splice consequence with junction changes and
            reading frame analysis.
        """
        from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import (
            SpliceEvent,
        )

        gene_name = gene or delta_result.gene
        variant_str = (
            f"{delta_result.chrom}:{delta_result.position} "
            f"{delta_result.ref}>{delta_result.alt}"
        )

        # Load gene structure if not provided
        if gene_structure is None and gene_name:
            gene_structure = self.get_gene_structure(gene_name)

        events = delta_result.events
        if not events:
            return SpliceConsequence(
                variant=variant_str,
                gene=gene_name,
                consequence_type=NO_EFFECT,
                confidence="LOW",
            )

        # Step 1: Map events to annotated splice sites
        annotated_events = self._map_events_to_annotation(events, gene_structure)

        # Step 2: Pair loss + gain events into junction changes
        junction_changes = self._pair_events(events, annotated_events, gene_structure)

        # Step 3: Classify consequence
        consequence_type = self._classify_consequence(events, annotated_events, junction_changes)

        # Step 4: Identify affected exons
        affected_exons = self._find_affected_exons(annotated_events, gene_structure)

        # Step 5: Reading frame analysis
        frame_preserved, protein_impact = self._analyze_reading_frame(
            junction_changes, gene_structure,
        )

        # Confidence from max delta magnitude
        max_delta = max(abs(e.delta) for e in events)
        if max_delta >= _HIGH_THRESHOLD:
            confidence = "HIGH"
        elif max_delta >= _MODERATE_THRESHOLD:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        # Build detail strings
        details = []
        for e in events[:10]:
            ann = annotated_events.get(id(e), {})
            site_type = ann.get("site_type", "unknown")
            details.append(
                f"{e.event_type} at {delta_result.chrom}:{e.position} "
                f"(Δ={e.delta:+.3f}, {e.distance_from_variant:+d}bp, {site_type})"
            )

        return SpliceConsequence(
            variant=variant_str,
            gene=gene_name,
            consequence_type=consequence_type,
            junction_changes=junction_changes,
            affected_exons=affected_exons,
            frame_preserved=frame_preserved,
            protein_impact=protein_impact,
            confidence=confidence,
            details=details,
        )

    def _map_events_to_annotation(
        self,
        events: List["SpliceEvent"],
        gene_structure: Optional[GeneStructure],
    ) -> Dict[int, dict]:
        """Map each event to the nearest annotated splice site.

        Returns a dict keyed by ``id(event)`` with metadata:
        - ``site_type``: "canonical", "cryptic", or "shifted"
        - ``nearest_annotated``: position of nearest annotated site
        - ``distance_to_annotated``: distance in bp
        """
        result: Dict[int, dict] = {}

        if gene_structure is None:
            for e in events:
                result[id(e)] = {"site_type": "unknown", "nearest_annotated": None, "distance_to_annotated": None}
            return result

        # Collect all annotated splice site positions
        donors = set(gene_structure.donor_positions())
        acceptors = set(gene_structure.acceptor_positions())
        all_sites = donors | acceptors

        for e in events:
            # Find nearest annotated site of the same type
            if "donor" in e.event_type:
                candidate_sites = donors
            else:
                candidate_sites = acceptors

            if not candidate_sites:
                result[id(e)] = {"site_type": "cryptic", "nearest_annotated": None, "distance_to_annotated": None}
                continue

            nearest = min(candidate_sites, key=lambda s: abs(s - e.position))
            dist = abs(e.position - nearest)

            if dist == 0 or dist <= 2:
                site_type = "canonical"
            elif dist <= 50:
                site_type = "shifted"
            else:
                site_type = "cryptic"

            result[id(e)] = {
                "site_type": site_type,
                "nearest_annotated": nearest,
                "distance_to_annotated": dist,
            }

        return result

    def _pair_events(
        self,
        events: List["SpliceEvent"],
        annotated: Dict[int, dict],
        gene_structure: Optional[GeneStructure],
    ) -> List[JunctionChange]:
        """Pair loss and gain events into junction changes.

        For each loss event, finds the strongest gain event of the same
        splice type (donor/donor or acceptor/acceptor) within
        ``max_pairing_distance``.  Unpaired losses become intron retention;
        unpaired gains become cryptic sites.
        """
        losses = [e for e in events if not e.is_gain]
        gains = [e for e in events if e.is_gain]
        junction_changes = []
        paired_gains = set()

        for loss in losses:
            meta = annotated.get(id(loss), {})
            # Find strongest compatible gain within distance
            best_gain = None
            best_score = 0.0
            for gain in gains:
                if gain.splice_type != loss.splice_type:
                    continue
                if id(gain) in paired_gains:
                    continue
                dist = abs(gain.position - loss.position)
                if dist > self.max_pairing_distance:
                    continue
                if abs(gain.delta) > best_score:
                    best_gain = gain
                    best_score = abs(gain.delta)

            if best_gain is not None:
                paired_gains.add(id(best_gain))
                gain_meta = annotated.get(id(best_gain), {})
                # Determine junction type based on annotation status
                if meta.get("site_type") == "canonical":
                    if gain_meta.get("site_type") in ("shifted", "canonical"):
                        jtype = "shift"
                    else:
                        jtype = "canonical_loss"
                else:
                    jtype = "canonical_loss"

                d_pos = loss.position if "donor" in loss.event_type else best_gain.position
                a_pos = best_gain.position if "acceptor" in best_gain.event_type else loss.position
                junction_changes.append(
                    JunctionChange(
                        junction_type=jtype,
                        donor_pos=d_pos if "donor" in loss.event_type else None,
                        acceptor_pos=a_pos if "acceptor" in loss.event_type else None,
                        delta_score=loss.delta,
                        ref_junction=meta.get("site_type") == "canonical",
                        alt_junction=True,
                        intron_size=abs(best_gain.position - loss.position) if best_gain else None,
                    )
                )
            else:
                # Unpaired loss → potential intron retention
                junction_changes.append(
                    JunctionChange(
                        junction_type="canonical_loss",
                        donor_pos=loss.position if "donor" in loss.event_type else None,
                        acceptor_pos=loss.position if "acceptor" in loss.event_type else None,
                        delta_score=loss.delta,
                        ref_junction=meta.get("site_type") == "canonical",
                        alt_junction=False,
                        intron_size=None,
                    )
                )

        # Unpaired gains → cryptic site activation
        for gain in gains:
            if id(gain) not in paired_gains:
                junction_changes.append(
                    JunctionChange(
                        junction_type="cryptic_gain",
                        donor_pos=gain.position if "donor" in gain.event_type else None,
                        acceptor_pos=gain.position if "acceptor" in gain.event_type else None,
                        delta_score=gain.delta,
                        ref_junction=False,
                        alt_junction=True,
                        intron_size=None,
                    )
                )

        return junction_changes

    def _classify_consequence(
        self,
        events: List["SpliceEvent"],
        annotated: Dict[int, dict],
        junction_changes: List[JunctionChange],
    ) -> str:
        """Classify the primary splice consequence from events and junctions."""
        if not events:
            return NO_EFFECT

        # Categorize events
        canonical_losses = []
        cryptic_gains = []
        shifted_gains = []
        donor_losses = []
        acceptor_losses = []
        donor_gains = []
        acceptor_gains = []

        for e in events:
            meta = annotated.get(id(e), {})
            site_type = meta.get("site_type", "unknown")

            if not e.is_gain:
                if site_type == "canonical":
                    canonical_losses.append(e)
                if "donor" in e.event_type:
                    donor_losses.append(e)
                else:
                    acceptor_losses.append(e)
            else:
                if site_type == "shifted":
                    shifted_gains.append(e)
                elif site_type == "cryptic":
                    cryptic_gains.append(e)
                if "donor" in e.event_type:
                    donor_gains.append(e)
                else:
                    acceptor_gains.append(e)

        # Classification rules (in priority order)
        has_canonical_donor_loss = any("donor" in e.event_type for e in canonical_losses)
        has_canonical_acceptor_loss = any("acceptor" in e.event_type for e in canonical_losses)

        # Exon skipping: both donor and acceptor loss at canonical sites
        if has_canonical_donor_loss and has_canonical_acceptor_loss:
            return EXON_SKIPPING

        # Donor/acceptor shift: canonical loss + nearby shifted gain
        if has_canonical_donor_loss and shifted_gains:
            return DONOR_SHIFT
        if has_canonical_acceptor_loss and shifted_gains:
            return ACCEPTOR_SHIFT

        # Destruction with cryptic activation
        if has_canonical_donor_loss and cryptic_gains:
            return DONOR_DESTRUCTION
        if has_canonical_acceptor_loss and cryptic_gains:
            return ACCEPTOR_DESTRUCTION

        # Intron retention: canonical loss with no compensatory gain
        if has_canonical_donor_loss and not donor_gains:
            return INTRON_RETENTION
        if has_canonical_acceptor_loss and not acceptor_gains:
            return INTRON_RETENTION

        # Cryptic exon: donor gain + acceptor gain both in intronic regions
        if donor_gains and acceptor_gains:
            return CRYPTIC_EXON

        # Multiple events with no clear pattern
        if len(events) > 2:
            return COMPLEX

        return COMPLEX if events else NO_EFFECT

    def _find_affected_exons(
        self,
        annotated_events: Dict[int, dict],
        gene_structure: Optional[GeneStructure],
    ) -> List[int]:
        """Find which exon numbers are affected by the events."""
        if gene_structure is None:
            return []

        affected = set()
        for eid, meta in annotated_events.items():
            nearest = meta.get("nearest_annotated")
            if nearest is not None:
                for exon in gene_structure.exons:
                    if abs(exon.start - nearest) <= 2 or abs(exon.end - nearest) <= 2:
                        affected.add(exon.exon_number)
        return sorted(affected)

    def _analyze_reading_frame(
        self,
        junction_changes: List[JunctionChange],
        gene_structure: Optional[GeneStructure],
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Analyze whether splice changes preserve the reading frame.

        Returns (frame_preserved, protein_impact) tuple.
        """
        if gene_structure is None or not gene_structure.has_cds:
            return None, None

        # TODO: Implement full reading frame analysis using CDS boundaries.
        # For now, return None to indicate CDS is available but analysis
        # is not yet implemented.
        return None, None

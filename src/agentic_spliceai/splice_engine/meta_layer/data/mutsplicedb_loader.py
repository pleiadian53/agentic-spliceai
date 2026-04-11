"""MutSpliceDB loader for experimentally validated splice variants.

MutSpliceDB provides RNA-seq validated splice-altering variants from
TCGA and other sources. Each record includes the observed splicing
effect (intron retention, exon skipping), enabling validation of both
delta magnitude AND predicted consequence type.

Data source: https://brb.nci.nih.gov/splicing/
Publication: PMID 33600011

Usage::

    loader = MutSpliceDBLoader("data/mutsplicedb/splice_sites_induced.tsv")
    for variant in loader.iter_variants():
        print(f"{variant.gene}: {variant.hgvs} → {variant.effect_type}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import polars as pl

logger = logging.getLogger(__name__)

# Regex to extract ref>alt from HGVS coding notation
# Matches: c.236+1G>A, c.1803+1G>C, c.2989-3C>G, c.*1234A>G
_HGVS_SNV_PATTERN = re.compile(r"c\.[\d\-\+\*]+([ACGT])>([ACGT])")

# Regex to extract position offset from HGVS
# c.236+1G>A → intronic +1 from exon boundary
# c.2989-3C>G → intronic -3 from exon boundary
_HGVS_OFFSET_PATTERN = re.compile(r"c\.(\d+)([+-])(\d+)")


@dataclass
class MutSpliceDBRecord:
    """Single experimentally validated splice variant from MutSpliceDB."""

    chrom: str
    position: int  # approximate genomic position (locus center)
    strand: str
    gene: str
    transcript_id: str
    hgvs: str  # e.g., NM_005957.5:c.236+1G>A
    ref_allele: str  # parsed from HGVS
    alt_allele: str  # parsed from HGVS
    effect_type: str  # "intron_retention", "exon_skipping"
    site_type: str  # "intron_retention_region", etc.
    evidence_source: str  # "GDC/TCGA"
    evidence_samples: str  # TCGA sample IDs
    confidence: str  # "medium", "high"

    @property
    def is_intron_retention(self) -> bool:
        return "intron_retention" in self.effect_type

    @property
    def is_exon_skipping(self) -> bool:
        return "exon_skipping" in self.effect_type

    @property
    def is_snv(self) -> bool:
        return len(self.ref_allele) == 1 and len(self.alt_allele) == 1

    def get_coordinate_key(self) -> str:
        return f"{self.chrom}:{self.position}"


def _parse_hgvs_alleles(hgvs: str) -> tuple[str, str]:
    """Extract ref and alt alleles from HGVS notation.

    Parameters
    ----------
    hgvs : str
        e.g., ``"NM_005957.5:c.236+1G>A"``

    Returns
    -------
    (ref, alt) : tuple of str
        Single-character alleles, or ("", "") if unparseable.
    """
    m = _HGVS_SNV_PATTERN.search(hgvs)
    if m:
        return m.group(1), m.group(2)
    return "", ""


class MutSpliceDBLoader:
    """Load MutSpliceDB parsed splice variants.

    Parameters
    ----------
    tsv_path : Path or str
        Path to ``splice_sites_induced.tsv`` (parsed MutSpliceDB data).
    raw_csv_path : Path or str, optional
        Path to raw MutSpliceDB CSV export (for additional fields).
    """

    def __init__(
        self,
        tsv_path: str | Path = "data/mutsplicedb/splice_sites_induced.tsv",
        raw_csv_path: Optional[str | Path] = None,
    ) -> None:
        self.tsv_path = Path(tsv_path)
        self.raw_csv_path = Path(raw_csv_path) if raw_csv_path else None
        self._records: Optional[List[MutSpliceDBRecord]] = None

    def load_all(self) -> List[MutSpliceDBRecord]:
        """Parse TSV and return all records with ref/alt extracted from HGVS.

        Results are cached after first load.
        """
        if self._records is not None:
            return self._records

        if not self.tsv_path.exists():
            raise FileNotFoundError(
                f"MutSpliceDB data not found: {self.tsv_path}\n"
                f"See data/mutsplicedb/README.md for download instructions."
            )

        df = pl.read_csv(str(self.tsv_path), separator="\t")
        records = []

        for row in df.iter_rows(named=True):
            hgvs = row.get("inducing_variant", "")
            ref, alt = _parse_hgvs_alleles(hgvs)

            if not ref or not alt:
                logger.debug("Skipping unparseable HGVS: %s", hgvs)
                continue

            records.append(MutSpliceDBRecord(
                chrom=row["chrom"],
                position=int(row["position"]),
                strand=row.get("strand", "+"),
                gene=row.get("gene", ""),
                transcript_id=row.get("transcript_id", ""),
                hgvs=hgvs,
                ref_allele=ref,
                alt_allele=alt,
                effect_type=row.get("effect_type", ""),
                site_type=row.get("site_type", ""),
                evidence_source=row.get("evidence_source", ""),
                evidence_samples=row.get("evidence_samples", ""),
                confidence=row.get("confidence", ""),
            ))

        self._records = records
        logger.info(
            "MutSpliceDB: loaded %d records from %s (%d skipped)",
            len(records), self.tsv_path.name, len(df) - len(records),
        )
        return records

    def get_intron_retention(self) -> List[MutSpliceDBRecord]:
        """Get intron retention variants."""
        return [r for r in self.load_all() if r.is_intron_retention]

    def get_exon_skipping(self) -> List[MutSpliceDBRecord]:
        """Get exon skipping variants."""
        return [r for r in self.load_all() if r.is_exon_skipping]

    def iter_variants(
        self,
        effect_type: Optional[str] = None,
        genes: Optional[List[str]] = None,
        chromosomes: Optional[List[str]] = None,
    ) -> Iterator[MutSpliceDBRecord]:
        """Iterate over filtered variants."""
        chrom_set = None
        if chromosomes:
            chrom_set = set()
            for c in chromosomes:
                chrom_set.add(c)
                chrom_set.add(c.replace("chr", "") if c.startswith("chr") else f"chr{c}")

        gene_set = set(genes) if genes else None

        for r in self.load_all():
            if effect_type and r.effect_type != effect_type:
                continue
            if gene_set and r.gene not in gene_set:
                continue
            if chrom_set and r.chrom not in chrom_set:
                continue
            yield r

    def get_statistics(self) -> Dict:
        """Summary statistics."""
        records = self.load_all()
        return {
            "total": len(records),
            "n_snvs": sum(1 for r in records if r.is_snv),
            "n_intron_retention": sum(1 for r in records if r.is_intron_retention),
            "n_exon_skipping": sum(1 for r in records if r.is_exon_skipping),
            "n_genes": len(set(r.gene for r in records)),
            "effect_types": {
                et: sum(1 for r in records if r.effect_type == et)
                for et in set(r.effect_type for r in records)
            },
            "confidence_distribution": {
                c: sum(1 for r in records if r.confidence == c)
                for c in set(r.confidence for r in records)
            },
        }

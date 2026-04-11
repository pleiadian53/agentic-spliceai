"""ClinVar VCF loader for variant effect benchmarking.

Parses ClinVar VCF files and provides filtered access to pathogenic
and benign variants for evaluating splice site prediction models.

Follows the same pattern as :class:`SpliceVarDBLoader` for consistency.

Usage::

    loader = ClinVarLoader("data/clinvar/clinvar_GRCh38.vcf.gz")
    stats = loader.get_statistics()
    print(f"Pathogenic: {stats['n_pathogenic']}, Benign: {stats['n_benign']}")

    for variant in loader.iter_variants(classification="Pathogenic", min_stars=1):
        print(f"{variant.chrom}:{variant.position} {variant.ref_allele}>{variant.alt_allele}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClinVarRecord:
    """Single variant record from ClinVar VCF."""

    clinvar_id: int
    chrom: str
    position: int  # 1-based
    ref_allele: str
    alt_allele: str
    gene: str
    classification: str  # "Pathogenic", "Benign", "VUS"
    review_stars: int  # 0-4
    disease: str
    variant_type: str  # "SNV", "Indel", "Other"
    molecular_consequence: str  # e.g., "splice_donor_variant", "missense_variant"
    raw_clnsig: str  # original CLNSIG value

    @property
    def is_pathogenic(self) -> bool:
        return self.classification == "Pathogenic"

    @property
    def is_benign(self) -> bool:
        return self.classification == "Benign"

    @property
    def is_vus(self) -> bool:
        return self.classification == "VUS"

    @property
    def is_snv(self) -> bool:
        return self.variant_type == "SNV"

    @property
    def is_splice_annotated(self) -> bool:
        """Whether ClinVar's molecular consequence mentions splicing."""
        return "splice" in self.molecular_consequence.lower()

    def get_coordinate_key(self) -> str:
        return f"{self.chrom}:{self.position}"


# ---------------------------------------------------------------------------
# Classification normalization
# ---------------------------------------------------------------------------

_PATHOGENIC_TERMS = {
    "Pathogenic",
    "Likely_pathogenic",
    "Pathogenic/Likely_pathogenic",
}

_BENIGN_TERMS = {
    "Benign",
    "Likely_benign",
    "Benign/Likely_benign",
}

# Review status → star count mapping
_REVIEW_STARS = {
    "no_assertion_criteria_provided": 0,
    "no_assertion_provided": 0,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "reviewed_by_expert_panel": 3,
    "practice_guideline": 4,
}


def _normalize_clnsig(clnsig: str) -> str:
    """Normalize ClinVar CLNSIG to Pathogenic/Benign/VUS."""
    if not clnsig:
        return "VUS"
    # Handle multi-value (e.g., "Pathogenic|risk_factor")
    primary = clnsig.split("|")[0].split(",")[0].strip()
    if primary in _PATHOGENIC_TERMS:
        return "Pathogenic"
    if primary in _BENIGN_TERMS:
        return "Benign"
    return "VUS"


def _parse_review_stars(revstat: str) -> int:
    """Convert CLNREVSTAT to star count."""
    if not revstat:
        return 0
    # Take the first status if multiple
    primary = revstat.split(",_")[0] if ",_" in revstat else revstat
    # Try exact match first, then prefix match
    for key, stars in _REVIEW_STARS.items():
        if revstat.startswith(key) or primary == key:
            return stars
    return 0


def _classify_variant_type(ref: str, alt: str) -> str:
    """Classify variant as SNV, Indel, or Other."""
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(ref) != len(alt):
        return "Indel"
    return "Other"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ClinVarLoader:
    """Load and parse ClinVar VCF for variant effect benchmarking.

    Parameters
    ----------
    vcf_path : Path or str
        Path to ClinVar VCF file (can be .vcf or .vcf.gz).
    build : str
        Genome build (default ``"GRCh38"``).
    """

    def __init__(self, vcf_path: str | Path, build: str = "GRCh38") -> None:
        self.vcf_path = Path(vcf_path)
        self.build = build
        self._records: Optional[List[ClinVarRecord]] = None

    def load_all(self) -> List[ClinVarRecord]:
        """Parse VCF and return all records.

        Results are cached after first load.
        """
        if self._records is not None:
            return self._records

        import pysam

        records = []
        vcf = pysam.VariantFile(str(self.vcf_path))

        for rec in vcf:
            chrom = rec.contig
            pos = rec.start + 1  # pysam is 0-based, we use 1-based

            # Handle multi-allelic: iterate ALT alleles
            for alt in rec.alts or []:
                info = rec.info

                # Extract fields with safe defaults
                clnsig = str(info.get("CLNSIG", ("",))[0]) if "CLNSIG" in info else ""
                revstat = str(info.get("CLNREVSTAT", ("",))[0]) if "CLNREVSTAT" in info else ""
                disease = str(info.get("CLNDN", ("",))[0]) if "CLNDN" in info else ""
                gene_info = str(info.get("GENEINFO", "")) if "GENEINFO" in info else ""
                mc_raw = str(info.get("MC", ("",))[0]) if "MC" in info else ""

                # Parse gene name from GENEINFO (format: "GENE:ID|GENE2:ID2")
                gene = ""
                if gene_info:
                    gene = gene_info.split(":")[0].split("|")[0]

                # Parse molecular consequence (format: "SO:0001575|splice_donor_variant")
                mol_consequence = ""
                if mc_raw and "|" in mc_raw:
                    mol_consequence = mc_raw.split("|", 1)[1]

                classification = _normalize_clnsig(clnsig)
                stars = _parse_review_stars(revstat)
                vtype = _classify_variant_type(rec.ref, alt)

                records.append(ClinVarRecord(
                    clinvar_id=rec.id if rec.id else 0,
                    chrom=chrom,
                    position=pos,
                    ref_allele=rec.ref,
                    alt_allele=alt,
                    gene=gene,
                    classification=classification,
                    review_stars=stars,
                    disease=disease,
                    variant_type=vtype,
                    molecular_consequence=mol_consequence,
                    raw_clnsig=clnsig,
                ))

        vcf.close()
        self._records = records
        logger.info("ClinVar: loaded %d records from %s", len(records), self.vcf_path.name)
        return records

    def get_pathogenic(self, min_stars: int = 0) -> List[ClinVarRecord]:
        """Get pathogenic/likely pathogenic variants."""
        return [
            r for r in self.load_all()
            if r.is_pathogenic and r.review_stars >= min_stars
        ]

    def get_benign(self, min_stars: int = 0) -> List[ClinVarRecord]:
        """Get benign/likely benign variants."""
        return [
            r for r in self.load_all()
            if r.is_benign and r.review_stars >= min_stars
        ]

    def get_snvs(
        self,
        classification: Optional[str] = None,
        min_stars: int = 0,
    ) -> List[ClinVarRecord]:
        """Get SNVs only, optionally filtered by classification."""
        records = self.load_all()
        return [
            r for r in records
            if r.is_snv
            and r.review_stars >= min_stars
            and (classification is None or r.classification == classification)
        ]

    def get_splice_relevant(
        self,
        gene_names: Optional[Set[str]] = None,
        splice_site_positions: Optional[Set[str]] = None,
        max_distance_to_splice: int = 50,
        min_stars: int = 0,
        snvs_only: bool = True,
    ) -> List[ClinVarRecord]:
        """Get variants likely to affect splicing.

        ClinVar classifies by disease pathogenicity, not by splice
        effect.  Most pathogenic variants are missense or frameshift —
        NOT splice-altering.  This method filters to variants near
        annotated splice sites where splice disruption is plausible.

        Filtering strategy (applied in order):
        1. Pathogenic or Benign classification (skip VUS)
        2. SNVs only (indels deferred)
        3. Gene filter (if provided)
        4. Near splice site (if positions provided)

        Parameters
        ----------
        gene_names : set of str, optional
            Filter to these genes.  If None, returns all.
        splice_site_positions : set of str, optional
            Set of ``"chrom:position"`` keys for known splice sites.
            If provided, only return variants within
            ``max_distance_to_splice`` bp of a splice site.
        max_distance_to_splice : int
            Maximum distance (bp) from a known splice site.
        min_stars : int
            Minimum review star count.
        snvs_only : bool
            If True, exclude indels.
        """
        records = self.load_all()
        result = []
        for r in records:
            if r.review_stars < min_stars:
                continue
            if snvs_only and not r.is_snv:
                continue
            if r.classification not in ("Pathogenic", "Benign"):
                continue
            if gene_names is not None and r.gene not in gene_names:
                continue
            if splice_site_positions is not None:
                near_splice = False
                for offset in range(-max_distance_to_splice, max_distance_to_splice + 1):
                    key = f"{r.chrom}:{r.position + offset}"
                    if key in splice_site_positions:
                        near_splice = True
                        break
                if not near_splice:
                    continue
            result.append(r)
        return result

    def iter_variants(
        self,
        classification: Optional[str] = None,
        min_stars: int = 0,
        chromosomes: Optional[List[str]] = None,
        snvs_only: bool = True,
    ) -> Iterator[ClinVarRecord]:
        """Iterate over filtered variants."""
        chrom_set = None
        if chromosomes:
            chrom_set = set()
            for c in chromosomes:
                chrom_set.add(c)
                chrom_set.add(c.replace("chr", "") if c.startswith("chr") else f"chr{c}")

        for r in self.load_all():
            if classification and r.classification != classification:
                continue
            if r.review_stars < min_stars:
                continue
            if snvs_only and not r.is_snv:
                continue
            if chrom_set and r.chrom not in chrom_set:
                continue
            yield r

    def get_statistics(self) -> Dict:
        """Summary statistics of the loaded data."""
        records = self.load_all()
        snvs = [r for r in records if r.is_snv]
        return {
            "total": len(records),
            "snvs": len(snvs),
            "n_pathogenic": sum(1 for r in records if r.is_pathogenic),
            "n_benign": sum(1 for r in records if r.is_benign),
            "n_vus": sum(1 for r in records if r.is_vus),
            "n_pathogenic_snvs": sum(1 for r in snvs if r.is_pathogenic),
            "n_benign_snvs": sum(1 for r in snvs if r.is_benign),
            "n_genes": len(set(r.gene for r in records if r.gene)),
            "stars_distribution": {
                i: sum(1 for r in records if r.review_stars == i)
                for i in range(5)
            },
        }

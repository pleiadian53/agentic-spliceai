"""Paralog and gene family detection for homology-aware data splitting.

Provides two strategies for identifying homologous genes:

1. **Sequence-based** (gold standard): Uses minimap2/mappy to align gene sequences
   and detect paralogs by identity/coverage thresholds. Requires ``mappy`` and
   gene sequences. This is what SpliceAI/OpenSpliceAI use.

2. **Name-based** (lightweight heuristic): Groups genes by shared root name
   (e.g., HOXA1/HOXA2 → family "HOX"). Fast, no external data needed,
   catches most obvious families. Useful as a complement or fallback.

Both approaches return gene family mappings that the splitting module uses
to ensure no family straddles train/test boundaries.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gene name-based family detection (lightweight heuristic)
# ---------------------------------------------------------------------------

# Known gene family prefixes in human genome. Genes matching these patterns
# are grouped into families. The regex captures the family root.
_FAMILY_PATTERNS: list[tuple[str, re.Pattern]] = [
    # HOX cluster: HOXA1, HOXB2, HOXC3, HOXD4 → family "HOX"
    ("HOX", re.compile(r"^(HOX)[A-D]\d+")),
    # Keratins: KRT1, KRT14, KRTAP1-1 → family "KRT"
    ("KRT", re.compile(r"^(KRT)\w+")),
    # Hemoglobins: HBA1, HBB, HBD → family "HB"
    ("HB", re.compile(r"^(HB)[A-Z]\d*$")),
    # Collagens: COL1A1, COL2A1 → family "COL"
    ("COL", re.compile(r"^(COL)\d+A\d+")),
    # Olfactory receptors: OR1A1, OR2B2 → family "OR"
    ("OR", re.compile(r"^(OR)\d+[A-Z]\d+")),
    # Zinc fingers: ZNF1, ZNF2 → family "ZNF"
    ("ZNF", re.compile(r"^(ZNF)\d+")),
    # Solute carriers: SLC1A1, SLC2A1 → family "SLC"
    ("SLC", re.compile(r"^(SLC)\d+A\d+")),
    # Protocadherins: PCDHA1, PCDHB2 → family "PCDH"
    ("PCDH", re.compile(r"^(PCDH)[A-Z]?\d+")),
    # UDP glucuronosyltransferases: UGT1A1, UGT2B7 → family "UGT"
    ("UGT", re.compile(r"^(UGT)\d+[A-Z]\d+")),
    # Cytochrome P450: CYP1A1, CYP2D6 → family "CYP"
    ("CYP", re.compile(r"^(CYP)\d+[A-Z]\d+")),
    # Defensins: DEFA1, DEFB1 → family "DEF"
    ("DEF", re.compile(r"^(DEF)[A-Z]\d+")),
    # Histones: H1-1, H2AC1, H3C1 → family "HIST"
    ("HIST", re.compile(r"^(H[1-4])[A-Z]?[A-Z]?\d+")),
    # Claudins: CLDN1, CLDN2 → family "CLDN"
    ("CLDN", re.compile(r"^(CLDN)\d+")),
    # Serpin: SERPINA1, SERPINB1 → family "SERPIN"
    ("SERPIN", re.compile(r"^(SERPIN)[A-Z]\d+")),
    # Immunoglobulin-like: IGLV1-1, IGHV1-1 → family "IG"
    ("IG", re.compile(r"^(IG[HKL])[VDJ]")),
    # T-cell receptors: TRAV1, TRBV1 → family "TR"
    ("TR", re.compile(r"^(TR[AB])[VDJ]")),
]

# Generic pattern: strip trailing digits from gene names to find family roots.
# E.g., BRCA1/BRCA2 → "BRCA", TP53 stays "TP53" (only 1 member → singleton).
_GENERIC_SUFFIX_RE = re.compile(r"^([A-Z]{2,}?)\d+$")


def detect_gene_families_by_name(
    gene_names: list[str],
    min_family_size: int = 2,
) -> Dict[str, str]:
    """Assign genes to families based on gene name patterns.

    Parameters
    ----------
    gene_names:
        List of gene symbols (e.g., ["BRCA1", "BRCA2", "HOXA1", "HOXA2", "TP53"]).
    min_family_size:
        Minimum number of members for a group to be considered a family.
        Singletons (genes with no detected relatives) get family=None.

    Returns
    -------
    Dict mapping gene_name -> family_id (str). Genes with no detected family
    are mapped to None (not included in the dict).
    """
    # Step 1: Try known family patterns
    gene_to_family: Dict[str, str] = {}
    family_members: Dict[str, List[str]] = defaultdict(list)

    gene_set = set(gene_names)

    for gene in gene_names:
        matched = False
        for family_id, pattern in _FAMILY_PATTERNS:
            if pattern.match(gene):
                gene_to_family[gene] = family_id
                family_members[family_id].append(gene)
                matched = True
                break

        if not matched:
            # Step 2: Generic suffix stripping
            m = _GENERIC_SUFFIX_RE.match(gene)
            if m:
                root = m.group(1)
                gene_to_family[gene] = root
                family_members[root].append(gene)

    # Step 3: Filter to families with >= min_family_size members
    result = {}
    for gene, family in gene_to_family.items():
        if len(family_members[family]) >= min_family_size:
            result[gene] = family

    n_families = len({f for f in result.values()})
    n_genes = len(result)
    logger.info(
        "Name-based family detection: %d genes in %d families (of %d total genes)",
        n_genes, n_families, len(gene_names),
    )

    return result


def get_paralog_groups(gene_to_family: Dict[str, str]) -> Dict[str, Set[str]]:
    """Invert the gene->family mapping to get family->gene_set.

    Parameters
    ----------
    gene_to_family:
        Output of ``detect_gene_families_by_name`` or similar.

    Returns
    -------
    Dict mapping family_id -> set of gene names in that family.
    """
    groups: Dict[str, Set[str]] = defaultdict(set)
    for gene, family in gene_to_family.items():
        groups[family].add(gene)
    return dict(groups)


# ---------------------------------------------------------------------------
# Sequence-based paralog detection (gold standard, requires mappy)
# ---------------------------------------------------------------------------


def detect_paralogs_by_alignment(
    train_sequences: Dict[str, str],
    test_sequences: Dict[str, str],
    min_identity: float = 0.8,
    min_coverage: float = 0.5,
) -> Set[str]:
    """Detect test genes that are paralogous to training genes via sequence alignment.

    Uses minimap2 (via mappy) to align each test gene sequence against the
    full training set. Test genes with any hit exceeding the identity/coverage
    thresholds are flagged as paralogous.

    This is the same approach used by OpenSpliceAI's ``paralogs.py``.

    Parameters
    ----------
    train_sequences:
        Dict mapping gene_id -> DNA sequence for all training genes.
    test_sequences:
        Dict mapping gene_id -> DNA sequence for all test genes.
    min_identity:
        Minimum alignment identity (matching bases / alignment length).
    min_coverage:
        Minimum query coverage (alignment length / query length).

    Returns
    -------
    Set of test gene IDs that have paralogs in the training set.

    Raises
    ------
    ImportError
        If ``mappy`` is not installed.
    """
    try:
        import mappy as mp
    except ImportError:
        raise ImportError(
            "mappy is required for sequence-based paralog detection: "
            "pip install mappy"
        )

    import tempfile

    # Write training sequences to temp FASTA for indexing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        for gene_id, seq in train_sequences.items():
            f.write(f">{gene_id}\n{seq}\n")
        temp_path = f.name

    try:
        aligner = mp.Aligner(temp_path, preset="map-ont")
        if not aligner:
            raise RuntimeError("Failed to build mappy index")

        paralogous_genes: Set[str] = set()
        total = len(test_sequences)

        for i, (gene_id, seq) in enumerate(test_sequences.items()):
            for hit in aligner.map(seq):
                identity = hit.mlen / hit.blen if hit.blen > 0 else 0
                coverage = hit.blen / len(seq) if len(seq) > 0 else 0
                if identity >= min_identity and coverage >= min_coverage:
                    paralogous_genes.add(gene_id)
                    break

            if (i + 1) % 500 == 0:
                logger.info(
                    "Paralog check: %d/%d test genes processed, %d flagged",
                    i + 1, total, len(paralogous_genes),
                )

        logger.info(
            "Sequence-based paralog detection: %d/%d test genes flagged (%.1f%%)",
            len(paralogous_genes), total,
            100 * len(paralogous_genes) / total if total else 0,
        )
        return paralogous_genes

    finally:
        Path(temp_path).unlink(missing_ok=True)

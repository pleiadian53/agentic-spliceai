"""Pydantic request/response models for AgenticSpliceAI Lab."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class GeneRecord(BaseModel):
    gene_id: str
    gene_name: str
    description: str = ''
    aliases: str = ''          # comma-joined gene synonyms, e.g. "ALS10,TDP-43"
    chrom: str
    strand: str
    start: int
    end: int
    length: int
    n_splice_sites: int = 0


class GeneListResponse(BaseModel):
    genes: List[GeneRecord]
    total: int
    page: int
    per_page: int
    total_pages: int


class GeneStatsResponse(BaseModel):
    model: str
    build: str
    annotation_source: str
    total_genes: int
    per_chromosome: Dict[str, int]


class ModelInfo(BaseModel):
    name: str
    build: str
    annotation_source: str


class MetricsRunInfo(BaseModel):
    run_id: str
    model: str
    build: str
    n_genes: int
    timestamp: Optional[str] = None
    path: str


class SpliceSiteMarker(BaseModel):
    position: int
    site_type: str        # 'donor' or 'acceptor'
    pred_type: str        # 'TP', 'FP', or 'FN'
    donor_score: float
    acceptor_score: float


class GenomeResponse(BaseModel):
    gene_name: str
    gene_id: str
    chrom: str
    strand: str
    gene_start: int
    gene_end: int
    model: str
    threshold: float
    positions: List[int]
    donor_prob: List[float]
    acceptor_prob: List[float]
    gt_positions: List[int]
    gt_site_types: List[str]
    markers: List[SpliceSiteMarker]
    n_tp: int
    n_fp: int
    n_fn: int
    downsample_factor: int
    total_positions: int

    # ── Meta-layer overlay (populated only when a meta model is requested) ──
    # When set, donor_prob/acceptor_prob/markers/n_* above are the BASE model's
    # (the exact OpenSpliceAI scores the meta layer refines), and these carry the
    # meta layer's prediction at the same positions — for a base-vs-meta overlay.
    meta_model: Optional[str] = None
    # A base model and the meta model refined from it have very different score
    # distributions, so one shared cutoff scores at least one of them at the wrong
    # operating point. Held-out F1-optima differ by more than 3x. Defaults to
    # ``threshold`` when the caller does not ask for a separate one.
    meta_threshold: Optional[float] = None
    meta_donor_prob: Optional[List[float]] = None
    meta_acceptor_prob: Optional[List[float]] = None
    meta_markers: Optional[List[SpliceSiteMarker]] = None
    meta_n_tp: Optional[int] = None
    meta_n_fp: Optional[int] = None
    meta_n_fn: Optional[int] = None


# ── Novel Site Explorer (M3) ──────────────────────────────────────────────────
# A ranker, not an overlay: M3 answers "which unannotated positions in this gene
# are worth inspecting?", so the payload is sparse ranked rows rather than the
# dense parallel arrays GenomeResponse carries.

class NovelSiteCandidate(BaseModel):
    rank: int
    position: int
    splice_type: str            # 'donor' | 'acceptor'
    strand: str
    meta_prob: float            # M3 score
    base_prob: float            # base model at the same position
    base_rank: Optional[int] = None   # rank under the base model, same filtering
    dinucleotide: str = ''      # GT/AG in transcript orientation; a coordinate sanity check

    # Independent evidence. Deliberately excludes SpliceVault / positives_pooled:
    # those are M3's own training pool, so showing them as support would be circular.
    longread_confirmed: bool = False
    longread_n_biosamples: Optional[int] = None
    disease_anchor: bool = False
    disease_mechanism: Optional[str] = None
    disease_source: Optional[str] = None


class NovelSitesResponse(BaseModel):
    gene_name: str
    gene_id: str
    chrom: str
    strand: str
    gene_start: int
    gene_end: int
    meta_model: str
    top_k: int
    min_prob: float
    candidates: List[NovelSiteCandidate]
    n_annotated_masked: int     # sites removed by the novelty post-filter
    n_considered: int           # positions scored (gene length)
    n_disease_anchors_in_gene: int


class NovelSiteGene(BaseModel):
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    has_disease_anchor: bool = False

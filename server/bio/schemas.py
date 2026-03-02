"""Pydantic request/response models for AgenticSpliceAI Lab."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class GeneRecord(BaseModel):
    gene_id: str
    gene_name: str
    description: str = ''
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

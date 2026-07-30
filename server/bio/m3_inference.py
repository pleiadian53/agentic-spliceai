"""M3 novel-site ranking for the Bio Lab UI.

M3 answers a different question from M1-S/M2-S, so it gets its own inference path
rather than reusing the base-vs-meta overlay:

    "For this gene, which **unannotated** positions are the best candidates to inspect?"

The output is a sparse ranked list, not a dense per-position overlay. Ranking,
the novelty post-filter, peak suppression, and channel selection all come from
``splice_engine.eval.novel_site_ranking`` so the UI and the offline evaluation
harness (``examples/meta_layer/13_evaluate_m3_novel.py``) agree by construction.

**Serving universe.** Genes are served from the M3 evaluation cache, whose universe
is the held-out SpliceAI test chromosomes (1/3/5/7/9). Every inspectable gene is one
the model never trained on. Genes outside it raise ``FileNotFoundError`` rather than
silently streaming bigWigs on the request path (the same stance as
``meta_inference``).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from agentic_spliceai.splice_engine.eval.novel_site_ranking import (
    IDX_TO_TYPE,
    GeneInterval,
    SiteEvidenceIndex,
    SiteIndex,
    load_sites_parquet,
    slice_mm_channels,
    suppress_adjacent,
    top_novel_candidates,
)
from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene
from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import _load_gene_npz
from agentic_spliceai.splice_engine.resources import get_meta_model_config

from . import config
from .meta_model_cache import get_meta_model_sync

logger = logging.getLogger(__name__)

# Complement table for reading a minus-strand dinucleotide off the plus-strand sequence.
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Inverse of IDX_TO_TYPE: 'donor' -> 0, 'acceptor' -> 1 (score-array column order).
TYPE_TO_IDX = {v: k for k, v in IDX_TO_TYPE.items()}


# ---------------------------------------------------------------------------
# Lazily-built, process-wide lookups (small tables, reused across requests)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _gene_table() -> pl.DataFrame:
    """The inspectable gene universe: one row per cached gene."""
    if not config.M3_EVAL_GENES.exists():
        raise FileNotFoundError(
            f"M3 gene universe not found at {config.M3_EVAL_GENES}. "
            "The Novel Site Explorer needs the M3 evaluation cache."
        )
    df = pl.read_parquet(config.M3_EVAL_GENES)
    # gene_id is 'gene-<NAME>' and matches the .npz filename; expose the bare name too.
    return df.with_columns(
        pl.col("gene_id").str.replace(r"^gene-", "").alias("gene_name")
    )


@lru_cache(maxsize=1)
def _annotation_index() -> Tuple[SiteIndex, int]:
    """Annotated sites (GENCODE ∪ RefSeq) — the novelty post-filter set."""
    df = load_sites_parquet(config.M3_ANNOTATION_MASK)
    return SiteIndex.from_df(df), df.height


@lru_cache(maxsize=1)
def _longread_evidence() -> SiteEvidenceIndex:
    """D1 — ENCODE long-read confirmed novel junctions, with biosample support."""
    df = load_sites_parquet(config.M3_LONGREAD_TRUTH, extra_cols=("n_biosamples",))
    return SiteEvidenceIndex.from_df(df, ["n_biosamples"])


@lru_cache(maxsize=1)
def _disease_evidence() -> Tuple[SiteEvidenceIndex, pl.DataFrame]:
    """D2 — held-out disease cryptic sites (novel subset), with mechanism/source."""
    df = load_sites_parquet(
        config.M3_DISEASE_ANCHORS,
        is_novel_only=True,
        extra_cols=("mechanism", "source", "support"),
    )
    return SiteEvidenceIndex.from_df(df, ["mechanism", "source", "support"]), df


def _resolve_gene(gene_name: str) -> GeneInterval:
    """Look up a gene's window in the cached universe (case-insensitive)."""
    table = _gene_table()
    hit = table.filter(pl.col("gene_name").str.to_uppercase() == gene_name.upper())
    if hit.height == 0:
        raise FileNotFoundError(
            f"'{gene_name}' is not in the Novel Site Explorer universe. That universe is "
            f"the {table.height:,} held-out genes on test chromosomes 1/3/5/7/9 — the genes "
            "M3 never trained on. Try TARDBP, CFTR, EGFR, MUTYH, or NPRL2."
        )
    row = hit.row(0, named=True)
    return GeneInterval(
        gene_id=str(row["gene_id"]),
        chrom=str(row["chrom"]),
        start=int(row["start"]),
        end=int(row["end"]),
        strand=str(row["strand"]),
    )


def _dinucleotide(sequence: str, gene: GeneInterval, position: int, splice_type: str) -> str:
    """Read the splice dinucleotide in **transcript** orientation.

    A correct donor reads ``GT`` and a correct acceptor ``AG`` once strand is
    resolved — the project's standard coordinate oracle. Surfacing it per candidate
    doubles as a live correctness check on the coordinate frame, and tells a user
    whether a *novel* candidate sits at a canonical dinucleotide.

    The offsets below were determined empirically by scanning ``[-4, +4]`` against
    the annotated sites in the mask (the technique the project uses for reconciling
    catalogue conventions) and then validated over **1,372 annotated sites across 60
    random genes: 0.988 canonical overall, 0.98-0.99 on both strands and both
    types**. Do not "simplify" them without re-running that check — the offset is a
    property of the annotation's position convention, not something to reason about
    from first principles.
    """
    i = position - gene.start
    # Sites read "forward" from the stored position; the other two read 3 bp back.
    forward = (gene.strand == "+" and splice_type == "donor") or (
        gene.strand == "-" and splice_type == "acceptor"
    )
    j = i if forward else i - 3
    if j < 0 or j > len(sequence) - 2:
        return ""
    dinuc = sequence[j : j + 2].upper()
    return dinuc if gene.strand == "+" else dinuc.translate(_COMPLEMENT)[::-1]


def _rank_one_type(
    probs_col: np.ndarray,
    gene: GeneInterval,
    splice_type: str,
    annotation: SiteIndex,
    top_k: int,
    min_prob: float,
) -> List[Tuple[int, float]]:
    """Rank one ``(chrom, strand, splice_type)`` group: novelty filter → floor → NMS.

    Peak suppression runs **inside** this per-type call on purpose: donors and
    acceptors are distinct biological events, so a strong donor must never suppress
    a real acceptor a few bp away. Callers merge the groups afterwards.
    """
    annotated = annotation.positions_in(
        gene.chrom, gene.strand, splice_type, gene.start, gene.end
    )
    # Over-fetch: the score floor and peak suppression both remove rows, and we
    # still want a full top-k afterwards.
    over_k = max(top_k * 10, 200)
    positions, scores = top_novel_candidates(probs_col, gene.start, annotated, over_k)

    keep = scores >= min_prob
    positions, scores = positions[keep], scores[keep]
    positions, scores = suppress_adjacent(
        positions, scores, min_separation=config.PEAK_MIN_SEPARATION
    )
    return list(zip(positions.tolist(), scores.tolist()))


def rank_novel_candidates(
    gene_name: str,
    top_k: int = config.DEFAULT_TOP_K,
    min_prob: float = config.DEFAULT_MIN_PROB,
) -> Dict[str, Any]:
    """Top-k novel candidates for one gene — M3-ranked, novelty-filtered, deduped.

    Raises
    ------
    FileNotFoundError
        If the gene is outside the cached (held-out) universe.
    """
    gene = _resolve_gene(gene_name)

    npz = config.M3_GENE_CACHE_DIR / f"{gene.gene_id}.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"No M3 feature cache for {gene.gene_id} at {npz}."
        )
    data = _load_gene_npz(npz)
    sequence = str(data["sequence"])

    model, cfg = get_meta_model_sync(config.M3_MODEL_NAME)
    spec = get_meta_model_config(config.M3_MODEL_NAME)

    # The cache is written with every channel so one cache serves all models; M3 was
    # trained without the junction channels and must be fed the matching slice.
    gene_data = {
        "sequence": data["sequence"],
        "base_scores": data["base_scores"],
        "mm_features": slice_mm_channels(
            np.asarray(data["mm_features"], dtype=np.float32),
            cfg,
            exclude_channels=spec.get("exclude_channels"),
        ),
    }
    meta_probs = infer_full_gene(
        model,
        gene_data,
        window_size=cfg.window_size,
        context_padding=cfg.effective_context_padding,
        device=None,
    )
    base_probs = np.asarray(data["base_scores"], dtype=np.float32)

    annotation, _ = _annotation_index()
    longread = _longread_evidence()
    disease, disease_df = _disease_evidence()

    # Rank each splice type independently, then merge — see _rank_one_type.
    merged: List[Dict[str, Any]] = []
    base_ranks: Dict[Tuple[int, str], int] = {}
    n_masked = 0
    for idx, splice_type in IDX_TO_TYPE.items():
        n_masked += int(
            annotation.positions_in(
                gene.chrom, gene.strand, splice_type, gene.start, gene.end
            ).size
        )
        for pos, score in _rank_one_type(
            meta_probs[:, idx], gene, splice_type, annotation, top_k, min_prob
        ):
            merged.append({"position": pos, "splice_type": splice_type, "meta_prob": score})
        # Base ranked through the identical path so the comparison is apples-to-apples.
        for r, (pos, _s) in enumerate(
            _rank_one_type(base_probs[:, idx], gene, splice_type, annotation, top_k, min_prob),
            start=1,
        ):
            base_ranks[(pos, splice_type)] = r

    merged.sort(key=lambda c: c["meta_prob"], reverse=True)
    merged = merged[:top_k]

    candidates: List[Dict[str, Any]] = []
    for rank, cand in enumerate(merged, start=1):
        pos, stype = cand["position"], cand["splice_type"]
        i = pos - gene.start
        lr = longread.lookup(gene.chrom, gene.strand, stype, pos)
        d2 = disease.lookup(gene.chrom, gene.strand, stype, pos)
        candidates.append(
            {
                "rank": rank,
                "position": pos,
                "splice_type": stype,
                "strand": gene.strand,
                "meta_prob": float(cand["meta_prob"]),
                "base_prob": float(base_probs[i, TYPE_TO_IDX[stype]]),
                "base_rank": base_ranks.get((pos, stype)),
                "dinucleotide": _dinucleotide(sequence, gene, pos, stype),
                "longread_confirmed": lr is not None,
                "longread_n_biosamples": int(lr["n_biosamples"]) if lr else None,
                "disease_anchor": d2 is not None,
                "disease_mechanism": d2.get("mechanism") if d2 else None,
                "disease_source": d2.get("source") if d2 else None,
            }
        )

    n_anchors = disease_df.filter(
        (pl.col("chrom") == gene.chrom)
        & (pl.col("position") >= gene.start)
        & (pl.col("position") < gene.end)
    ).height

    return {
        "gene_name": gene.gene_id.replace("gene-", ""),
        "gene_id": gene.gene_id,
        "chrom": gene.chrom,
        "strand": gene.strand,
        "gene_start": gene.start,
        "gene_end": gene.end,
        "meta_model": config.M3_MODEL_NAME,
        "top_k": top_k,
        "min_prob": min_prob,
        "candidates": candidates,
        "n_annotated_masked": n_masked,
        "n_considered": int(len(sequence)),
        "n_disease_anchors_in_gene": n_anchors,
    }


def list_inspectable_genes() -> List[Dict[str, Any]]:
    """The gene universe for the picker, flagged with whether a D2 anchor falls inside."""
    genes = _gene_table()
    _, anchors = _disease_evidence()

    # disease_anchors has no gene column, so attach by coordinate containment.
    joined = (
        genes.join(anchors.select("chrom", "position"), on="chrom", how="left")
        .filter(
            pl.col("position").is_not_null()
            & (pl.col("position") >= pl.col("start"))
            & (pl.col("position") < pl.col("end"))
        )
        .select("gene_id")
        .unique()
    )
    with_anchor = set(joined.get_column("gene_id").to_list())

    return [
        {
            "gene_id": r["gene_id"],
            "gene_name": r["gene_name"],
            "chrom": r["chrom"],
            "strand": r["strand"],
            "has_disease_anchor": r["gene_id"] in with_anchor,
        }
        for r in genes.iter_rows(named=True)
    ]

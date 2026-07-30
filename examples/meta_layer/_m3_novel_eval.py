"""Library for the M3 novel-splice-site evaluation (Phase D / Tier 0).

Pure functions — **no model / bigWig / torch dependencies** — so the metric
math (the novelty post-filter + per-gene precision@k / recall@k against an
independent truth set) can be unit-tested on a hand-built fixture. The driver
``13_evaluate_m3_novel.py`` imports these.

Design decisions baked in here (see ``examples/meta_layer/docs/M3/m3_design.md``):

* **Novelty is applied, not scored.** Before ranking, a gene's scored positions
  are anti-joined against the annotation union (GENCODE ∪ RefSeq-curated); only
  the survivors are "novel candidates". Without this, every model just ranks the
  canonical annotated donors at the top and precision@k means nothing.
* **Coordinate frame.** A model's per-position output array indexes the
  forward-strand gene window ``[gene_start, gene_end)``; array index ``i`` maps to
  genomic position ``gene_start + i`` — the same frame ``build_m3_labels`` used to
  place training labels, so the genomic join to truth is training-consistent.
* **Join key = 4-tuple** ``(chrom, position, strand, splice_type)`` with a bare
  ``chrom`` (``"1"``, not ``"chr1"``). A donor and an acceptor can share a
  coordinate; strand is load-bearing for overlapping antisense genes. Predicted
  sites are assigned the gene's strand. (Training's label builder was
  strand-agnostic within the window; evaluating strand-strict is the
  biologically-correct, more conservative choice — documented, not accidental.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

# The ranking primitives live in the installed package so the Bio Lab UI can reuse
# them (a server module cannot import from examples/). They are re-exported here so
# this module's public API — and run_selftest() below, which guards them — is
# unchanged for the driver scripts.
from agentic_spliceai.splice_engine.eval.novel_site_ranking import (  # noqa: F401
    ACCEPTOR_IDX,
    DONOR_IDX,
    IDX_TO_TYPE,
    JOIN_KEY,
    GeneInterval,
    SiteEvidenceIndex,
    SiteIndex,
    _strip_chr_expr,
    load_sites_parquet,
    slice_mm_channels,
    strip_chr,
    suppress_adjacent,
    top_novel_candidates,
)


def _positions_by_chrom(truth_dfs: Sequence[pl.DataFrame]) -> Dict[str, np.ndarray]:
    """Union of truth positions per bare chrom, as sorted numpy arrays."""
    frames = [df.select("chrom", "position") for df in truth_dfs if df.height]
    if not frames:
        return {}
    allpos = pl.concat(frames).unique()
    out: Dict[str, np.ndarray] = {}
    for chrom, sub in allpos.group_by("chrom"):
        key = chrom[0] if isinstance(chrom, tuple) else chrom
        out[str(key)] = np.sort(sub.get_column("position").to_numpy())
    return out


def resolve_truth_genes(
    genes: Sequence[GeneInterval],
    truth_dfs: Sequence[pl.DataFrame],
) -> List[GeneInterval]:
    """Keep genes whose ``[start, end)`` window contains >=1 truth position.

    precision@k is only defined where truth exists, so this is the meaningful
    eval universe. Uses ``searchsorted`` — O(log n) per gene.
    """
    pos_by_chrom = _positions_by_chrom(truth_dfs)
    kept: List[GeneInterval] = []
    for g in genes:
        arr = pos_by_chrom.get(g.chrom)
        if arr is None or arr.size == 0:
            continue
        lo = np.searchsorted(arr, g.start, side="left")
        hi = np.searchsorted(arr, g.end, side="left")  # [start, end)
        if hi > lo:
            kept.append(g)
    return kept


# ---------------------------------------------------------------------------
# Per-gene scoring -> novel candidates -> precision@k / recall@k
# ---------------------------------------------------------------------------

@dataclass
class KMetricAccumulator:
    """Accumulates per-gene precision@k / recall@k, reports macro + micro."""

    ks: Tuple[int, ...] = (5, 10, 20)
    # macro sums (mean over genes); micro sums (pooled hits / pooled denom)
    _prec_sum: Dict[int, float] = None
    _rec_sum: Dict[int, float] = None
    _n_genes: int = 0
    _micro_hits: Dict[int, int] = None
    _micro_k: Dict[int, int] = None
    _micro_truth: int = 0

    def __post_init__(self):
        self._prec_sum = {k: 0.0 for k in self.ks}
        self._rec_sum = {k: 0.0 for k in self.ks}
        self._micro_hits = {k: 0 for k in self.ks}
        self._micro_k = {k: 0 for k in self.ks}

    def add_gene(self, ranked_positions: np.ndarray, truth_positions: np.ndarray) -> None:
        """One gene/type: ranked novel candidate positions vs truth positions."""
        n_truth = int(truth_positions.size)
        if n_truth == 0:
            return  # precision@k undefined without truth in this gene/type
        truth_set = set(int(x) for x in truth_positions)
        self._n_genes += 1
        self._micro_truth += n_truth
        for k in self.ks:
            topk = ranked_positions[:k]
            hits = sum(1 for x in topk.tolist() if x in truth_set)
            denom = min(k, ranked_positions.size) if ranked_positions.size else k
            self._prec_sum[k] += hits / max(1, denom)
            self._rec_sum[k] += hits / n_truth
            self._micro_hits[k] += hits
            self._micro_k[k] += denom

    def result(self) -> dict:
        n = max(1, self._n_genes)
        return {
            "n_genes_with_truth": self._n_genes,
            "n_truth_sites": self._micro_truth,
            "macro_precision_at_k": {k: self._prec_sum[k] / n for k in self.ks},
            "macro_recall_at_k": {k: self._rec_sum[k] / n for k in self.ks},
            "micro_precision_at_k": {
                k: (self._micro_hits[k] / self._micro_k[k]) if self._micro_k[k] else 0.0
                for k in self.ks
            },
            "micro_recall_at_k": {
                k: (self._micro_hits[k] / self._micro_truth) if self._micro_truth else 0.0
                for k in self.ks
            },
        }


def evaluate_gene(
    probs: np.ndarray,
    gene: GeneInterval,
    annotation: SiteIndex,
    truth: SiteIndex,
    accs: Dict[str, KMetricAccumulator],
    max_k: int,
) -> None:
    """Score one gene for both splice types, update the per-type accumulators.

    ``probs`` is ``[L, 3]`` (donor, acceptor, neither) aligned to
    ``[gene_start, gene_end)``. ``accs`` has keys ``"donor"``/``"acceptor"``.
    """
    for idx, stype in IDX_TO_TYPE.items():
        ann_pos = annotation.positions_in(gene.chrom, gene.strand, stype, gene.start, gene.end)
        ranked_pos, _ = top_novel_candidates(probs[:, idx], gene.start, ann_pos, max_k)
        truth_pos = truth.positions_in(gene.chrom, gene.strand, stype, gene.start, gene.end)
        accs[stype].add_gene(ranked_pos, truth_pos)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_comparison_table(per_model: Dict[str, dict], ks=(5, 10, 20)) -> str:
    """Compact per-model precision@k / recall@k table (macro, donor+acceptor combined)."""
    lines = []
    header = f"{'model':<22} " + " ".join(f"P@{k:<5} R@{k:<5}" for k in ks)
    lines.append(header)
    lines.append("-" * len(header))
    for name, res in per_model.items():
        # combine donor + acceptor via micro pooling for the headline row
        cells = []
        for k in ks:
            p = res["combined"]["macro_precision_at_k"][k]
            r = res["combined"]["macro_recall_at_k"][k]
            cells.append(f"{p:<6.3f} {r:<6.3f}")
        lines.append(f"{name:<22} " + " ".join(cells))
    return "\n".join(lines)


def combine_type_results(donor_res: dict, acceptor_res: dict, ks=(5, 10, 20)) -> dict:
    """Pool donor + acceptor accumulator results into one 'combined' summary."""
    ng = donor_res["n_genes_with_truth"] + acceptor_res["n_genes_with_truth"]
    nt = donor_res["n_truth_sites"] + acceptor_res["n_truth_sites"]
    out = {"n_genes_with_truth": ng, "n_truth_sites": nt,
           "macro_precision_at_k": {}, "macro_recall_at_k": {}}
    for k in ks:
        # weight each type's macro mean by its gene count
        dn, an = donor_res["n_genes_with_truth"], acceptor_res["n_genes_with_truth"]
        tot = max(1, dn + an)
        out["macro_precision_at_k"][k] = (
            donor_res["macro_precision_at_k"][k] * dn
            + acceptor_res["macro_precision_at_k"][k] * an
        ) / tot
        out["macro_recall_at_k"][k] = (
            donor_res["macro_recall_at_k"][k] * dn
            + acceptor_res["macro_recall_at_k"][k] * an
        ) / tot
    return out


# ---------------------------------------------------------------------------
# Self-test fixture (run: python _m3_novel_eval.py)
# ---------------------------------------------------------------------------

def run_selftest() -> int:
    """Prove the post-filter + precision@k/recall@k math on a hand-built fixture.

    No models, no bigWigs — pure array/dataframe logic.
    """
    import sys

    print("M3 novel-eval self-test (post-filter + precision@k math)")

    # ── Fixture gene: chrom '7', [100, 130), + strand, 30 positions ──────
    gene = GeneInterval(gene_id="FIX1", chrom="7", start=100, end=130, strand="+")
    L = gene.end - gene.start

    # Construct a donor-probability profile with known peaks.
    #   pos 105 (rel 5): prob 0.90  -> ANNOTATED (must be filtered out)
    #   pos 110 (rel 10): prob 0.80 -> TRUTH novel  (should be a hit)
    #   pos 118 (rel 18): prob 0.70 -> not truth, not annotated (false positive)
    #   pos 122 (rel 22): prob 0.60 -> TRUTH novel  (should be a hit)
    probs = np.zeros((L, 3), dtype=np.float32)
    probs[:, 2] = 1.0  # neither
    for rel, dp in [(5, 0.90), (10, 0.80), (18, 0.70), (22, 0.60)]:
        probs[rel, 0], probs[rel, 2] = dp, 1 - dp

    annotation = pl.DataFrame({
        "chrom": ["7"], "position": [105], "strand": ["+"], "splice_type": ["donor"],
    })
    truth = pl.DataFrame({
        "chrom": ["7", "7"], "position": [110, 122],
        "strand": ["+", "+"], "splice_type": ["donor", "donor"],
    })

    accs = {"donor": KMetricAccumulator(ks=(1, 2, 3)),
            "acceptor": KMetricAccumulator(ks=(1, 2, 3))}
    evaluate_gene(probs, gene, SiteIndex.from_df(annotation), SiteIndex.from_df(truth),
                  accs, max_k=3)
    donor = accs["donor"].result()

    ok = True

    def check(label, got, want):
        nonlocal ok
        good = abs(got - want) < 1e-9
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: got {got:.4f}, want {want:.4f}")

    # Post-filter must drop the annotated pos 105 (top prob 0.90). Ranked novel
    # donors: 110 (0.80), 118 (0.70), 122 (0.60). Truth = {110, 122}.
    #   P@1: top-1 = {110} -> 1 hit / 1 = 1.000
    #   P@2: {110,118} -> 1 hit / 2 = 0.500
    #   P@3: {110,118,122} -> 2 hits / 3 = 0.667
    check("precision@1", donor["macro_precision_at_k"][1], 1.0)
    check("precision@2", donor["macro_precision_at_k"][2], 0.5)
    check("precision@3", donor["macro_precision_at_k"][3], 2.0 / 3.0)
    #   R@1: 1/2 = 0.5 ; R@2: 1/2 ; R@3: 2/2 = 1.0
    check("recall@1", donor["macro_recall_at_k"][1], 0.5)
    check("recall@3", donor["macro_recall_at_k"][3], 1.0)
    print(f"  n_genes_with_truth={donor['n_genes_with_truth']} "
          f"n_truth_sites={donor['n_truth_sites']} (expect 1, 2)")
    ok = ok and donor["n_genes_with_truth"] == 1 and donor["n_truth_sites"] == 2

    # ── Post-filter unit check: annotated pos must be gone ───────────────
    ranked, _ = top_novel_candidates(probs[:, 0], gene.start, np.array([105]), max_k=5)
    ok = ok and 105 not in set(ranked.tolist())
    print(f"  [{'PASS' if 105 not in set(ranked.tolist()) else 'FAIL'}] "
          f"annotated pos 105 removed from ranking (ranked={ranked.tolist()})")

    # ── Gene-universe resolver check ─────────────────────────────────────
    g_in = GeneInterval("A", "7", 100, 130, "+")   # contains truth 110,122
    g_out = GeneInterval("B", "7", 200, 230, "+")  # no truth
    kept = resolve_truth_genes([g_in, g_out], [truth])
    ok = ok and [g.gene_id for g in kept] == ["A"]
    print(f"  [{'PASS' if [g.gene_id for g in kept] == ['A'] else 'FAIL'}] "
          f"universe resolver kept {[g.gene_id for g in kept]} (expect ['A'])")

    print(f"\nSELFTEST {'GREEN' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_selftest())

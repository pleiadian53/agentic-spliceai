"""How much of M3's novel-site ranking is a proximity-to-annotation prior?

The M3 positives sit close to annotated splice sites (median 49 nt; 96% within 500 nt), so a
ranker that only knows "GT/AG motif, near a known site" may already recover most of what the
evaluation rewards. This probe measures that, on the same held-out eval universe and truth sets
as ``01_evaluate_m3_novel.py``.

For every truth-containing gene and splice type it ranks the novel candidates three ways:

* ``base``: the OpenSpliceAI score (the base-model channel).
* ``M3-v1``: the M3 recognizer, scored from the cached dense gene windows.
* ``dist_motif``: a cis-only baseline. Candidates are non-annotated positions that carry the
  right GT/AG dinucleotide, ordered by distance to the nearest annotated site of the same type
  (closest first, ties broken at random).

It writes the per-gene precision@k / recall@k table (so the first two rows can be checked against
the published Tier 0 numbers), plus two parquet tables that feed a distance-stratified summary:
recall@20 per distance bin and the composition of each model's top 20.

Requires the M3 eval cache (``output/meta_layer/m3_eval_d1/``). Run (after
``mamba activate agentic-spliceai``)::

    python examples/meta_layer/m3/02_m3_distance_baseline_probe.py
    python examples/meta_layer/m3/02_m3_distance_baseline_probe.py --limit 100   # smoke test
    python examples/meta_layer/m3/02_m3_distance_baseline_probe.py --summarize-only

Outputs (default ``output/meta_layer/m3_distance_probe/``): ``metrics.json``, ``cand.parquet``,
``truth.parquet``, ``summary.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _m3_novel_eval as m3eval  # noqa: E402

log = logging.getLogger(__name__)

EVAL_DIR = REPO / "output/meta_layer/m3_eval_d1"
DEFAULT_OUT = REPO / "output/meta_layer/m3_distance_probe"
KS = (5, 10, 20)
MAX_K = max(KS)

# (strand, splice_type) -> (dinucleotide, offset of its first base from the site coordinate).
# The cached sequence is forward-strand genomic, so minus-strand sites carry the reverse
# complement (donor GT -> AC, acceptor AG -> CT). Verified: 100% of M3 positives match.
MOTIF = {
    ("+", "donor"): ("GT", 0),
    ("+", "acceptor"): ("AG", -3),
    ("-", "acceptor"): ("CT", 0),
    ("-", "donor"): ("AC", -3),
}

# Distance bins (nt to the nearest annotated site of the same type) for the stratified summary.
BIN_EDGES = [10, 50, 100, 500, 1000]
BIN_LABELS = ["<=10", "11-50", "51-100", "101-500", "501-1000", ">1000"]


def _load_eval13():
    """Import ``01_evaluate_m3_novel.py`` (digit-prefixed, so not importable by name)."""
    spec = importlib.util.spec_from_file_location("eval13", HERE / "01_evaluate_m3_novel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def motif_mask(seq_arr: np.ndarray, dinuc: str, off: int) -> np.ndarray:
    """Boolean mask: site coordinate ``i`` has ``dinuc`` at ``seq[i+off : i+off+2]``.

    Args:
        seq_arr: Upper-case sequence as a uint8 array.
        dinuc: Two-letter dinucleotide to look for.
        off: Offset of the dinucleotide's first base from the site coordinate.

    Returns:
        Boolean array of the same length as ``seq_arr``.
    """
    length = seq_arr.size
    pair = np.zeros(length, dtype=bool)
    pair[:-1] = (seq_arr[:-1] == ord(dinuc[0])) & (seq_arr[1:] == ord(dinuc[1]))
    mask = np.zeros(length, dtype=bool)
    if off >= 0:
        mask[: length - off] = pair[off:]
    else:
        mask[-off:] = pair[: length + off]
    return mask


def nearest_dist(pos: np.ndarray, ann: np.ndarray) -> np.ndarray:
    """Distance from each position to the nearest value in the sorted array ``ann``."""
    if ann.size == 0:
        return np.full(pos.shape, 10**9, dtype=np.int64)
    i = np.searchsorted(ann, pos)
    left = np.abs(pos - ann[np.clip(i - 1, 0, ann.size - 1)])
    right = np.abs(ann[np.clip(i, 0, ann.size - 1)] - pos)
    return np.minimum(left, right)


def run_probe(args: argparse.Namespace) -> None:
    """Score genes with each ranker and write the candidate / truth tables and metrics."""
    from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz,
    )
    from agentic_spliceai.splice_engine.utils.device import resolve_device

    eval13 = _load_eval13()
    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    uni = pl.read_parquet(EVAL_DIR / "eval_genes.parquet")
    genes = [
        m3eval.GeneInterval(r["gene_id"], r["chrom"], int(r["start"]), int(r["end"]), r["strand"])
        for r in uni.iter_rows(named=True)
    ]
    if args.limit:
        genes = genes[: args.limit]
    chroms = sorted({g.chrom for g in genes})

    load = m3eval.load_sites_parquet
    annotation = m3eval.SiteIndex.from_df(load(eval13.ANNOTATION_MASK, keep_chroms_bare=chroms))
    truth = {
        "D1": m3eval.SiteIndex.from_df(load(eval13.D1_TRUTH, keep_chroms_bare=chroms)),
        "D1_hiconf": m3eval.SiteIndex.from_df(
            load(eval13.D1_TRUTH, keep_chroms_bare=chroms, min_biosamples=2)
        ),
        "D2": m3eval.SiteIndex.from_df(
            load(eval13.D2_ANCHORS, keep_chroms_bare=chroms, is_novel_only=True)
        ),
    }
    specs = [s for s in eval13.DEFAULT_MODELS if s[0] in ("base", "M3-v1")]
    models = eval13._load_models(specs, device)
    names = list(models) + ["dist_motif"]

    accs = {
        ts: {m: {t: m3eval.KMetricAccumulator(KS) for t in ("donor", "acceptor")} for m in names}
        for ts in truth
    }
    cand_rows: list[tuple] = []
    truth_rows: list[tuple] = []
    t0 = time.time()
    for gi, gene in enumerate(genes):
        npz = EVAL_DIR / "gene_cache" / f"{gene.gene_id}.npz"
        if not npz.exists():
            continue
        data = _load_gene_npz(npz)
        if len(data["sequence"]) != gene.end - gene.start:
            continue
        seq_arr = np.frombuffer(data["sequence"].upper().encode(), dtype=np.uint8)
        base, mm9 = data["base_scores"], data["mm_features"]
        probs = {}
        for name, (model, cfg, transform) in models.items():
            if model is None:
                probs[name] = base
            else:
                gd = {"sequence": data["sequence"], "base_scores": base,
                      "mm_features": transform(mm9)}
                probs[name] = infer_full_gene(
                    model, gd, context_padding=cfg.effective_context_padding, device=device
                )
        rng = np.random.default_rng(sum(map(ord, gene.gene_id)))
        for idx, stype in m3eval.IDX_TO_TYPE.items():
            ann_pos = annotation.positions_in(gene.chrom, gene.strand, stype, gene.start, gene.end)
            tpos = {
                ts: truth[ts].positions_in(gene.chrom, gene.strand, stype, gene.start, gene.end)
                for ts in truth
            }
            if not any(p.size for p in tpos.values()):
                continue
            ranked: dict[str, tuple] = {}
            for name in models:
                ranked[name] = m3eval.top_novel_candidates(
                    probs[name][:, idx], gene.start, ann_pos, MAX_K
                )
            dinuc, off = MOTIF[(gene.strand, stype)]
            motif_ok = motif_mask(seq_arr, dinuc, off)
            cand = np.flatnonzero(motif_ok) + gene.start
            cand = cand[~np.isin(cand, ann_pos)]
            dist = nearest_dist(cand, ann_pos)
            order = np.lexsort((rng.random(cand.size), dist))[:MAX_K]
            ranked["dist_motif"] = (cand[order], -dist[order].astype(float))

            for ts in truth:
                if tpos[ts].size:
                    for name in names:
                        accs[ts][name][stype].add_gene(ranked[name][0], tpos[ts])

            d1, d1h, d2 = (set(tpos[k].tolist()) for k in ("D1", "D1_hiconf", "D2"))
            for name in names:
                pos, score = ranked[name]
                cdist = nearest_dist(pos, ann_pos)
                for rank, (p, s, d) in enumerate(zip(pos.tolist(), score.tolist(), cdist.tolist(), strict=True), 1):
                    cand_rows.append((gene.gene_id, stype, name, rank, p, s, d,
                                      p in d1, p in d1h, p in d2))
            all_truth = np.unique(np.concatenate(list(tpos.values())))
            tdist = nearest_dist(all_truth, ann_pos)
            for p, d in zip(all_truth.tolist(), tdist.tolist(), strict=True):
                truth_rows.append((gene.gene_id, stype, p, d, p in d1, p in d1h, p in d2,
                                   bool(motif_ok[p - gene.start])))
        if (gi + 1) % 100 == 0:
            log.info("%d/%d genes, %.0fs", gi + 1, len(genes), time.time() - t0)

    cand_cols = ["gene", "stype", "model", "rank", "pos", "score", "dist", "d1", "d1h", "d2"]
    truth_cols = ["gene", "stype", "pos", "dist", "d1", "d1h", "d2", "motif_ok"]
    pl.DataFrame(cand_rows, schema=cand_cols, orient="row").write_parquet(args.out / "cand.parquet")
    pl.DataFrame(truth_rows, schema=truth_cols, orient="row").write_parquet(
        args.out / "truth.parquet"
    )

    results = {}
    for ts in truth:
        per_model = {}
        for m in names:
            dn = accs[ts][m]["donor"].result()
            ac = accs[ts][m]["acceptor"].result()
            per_model[m] = {"combined": m3eval.combine_type_results(dn, ac, KS)}
        results[ts] = per_model
        print(f"\n{ts}: per-gene precision@k / recall@k (macro)")
        print(m3eval.format_comparison_table(per_model, KS))
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, default=str))


def _add_bin(df: pl.DataFrame) -> pl.DataFrame:
    """Add integer column ``bin`` (index into ``BIN_LABELS``) from the ``dist`` column."""
    bins = np.digitize(df["dist"].to_numpy(), BIN_EDGES, right=True)
    return df.with_columns(pl.Series("bin", bins))


def summarize(out_dir: Path) -> str:
    """Distance-stratified summary from ``cand.parquet`` and ``truth.parquet``.

    Two views per truth set. Recall@20 by *truth-site* distance bin: of the truth sites that sit
    in a given distance bin, what fraction appears in each model's top 20 (pooled over genes).
    Top-20 composition by *candidate* distance bin: how many of each model's slots land in the
    bin, and what fraction of them are truth.

    Returns:
        The summary as markdown (also written to ``summary.md``).
    """
    cand = pl.read_parquet(out_dir / "cand.parquet")
    truth = _add_bin(pl.read_parquet(out_dir / "truth.parquet"))
    cand = _add_bin(cand)
    models = sorted(cand["model"].unique().to_list())
    key = ["gene", "stype", "pos"]
    lines = ["# M3 distance-baseline probe: distance-stratified summary", ""]
    for tname, tcol in (("D1", "d1"), ("D1_hiconf", "d1h"), ("D2", "d2")):
        t = truth.filter(pl.col(tcol))
        if t.height == 0:
            continue
        lines += [f"## {tname}: recall@20 by truth-site distance bin (pooled)", "",
                  "| distance (nt) | n truth | " + " | ".join(models) + " |",
                  "|---|---:|" + "---:|" * len(models)]
        for b, label in enumerate(BIN_LABELS):
            tb = t.filter(pl.col("bin") == b)
            if tb.height == 0:
                continue
            cells = []
            for m in models:
                hits = cand.filter(pl.col("model") == m).select(key).unique().with_columns(
                    pl.lit(True).alias("_hit")
                )
                frac = tb.join(hits, on=key, how="left")["_hit"].fill_null(False).mean()
                cells.append(f"{frac:.3f}")
            lines.append(f"| {label} | {tb.height:,} | " + " | ".join(cells) + " |")
        lines += ["", f"## {tname}: top-20 composition by candidate distance bin", "",
                  "| model | distance (nt) | share of slots | precision |", "|---|---|---:|---:|"]
        for m in models:
            cm = cand.filter(pl.col("model") == m)
            for b, label in enumerate(BIN_LABELS):
                cb = cm.filter(pl.col("bin") == b)
                if cb.height:
                    lines.append(
                        f"| {m} | {label} | {cb.height / cm.height:.3f} | {cb[tcol].mean():.3f} |"
                    )
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N genes.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps.")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Skip scoring; rebuild summary.md from existing parquet tables.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.summarize_only:
        run_probe(args)
    print(summarize(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

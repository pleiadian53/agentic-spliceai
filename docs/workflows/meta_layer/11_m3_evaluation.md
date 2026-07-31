# Stage 11 — M3 Evaluation (Anti-Circular)

**Pipeline position:** `10 training → ` **this stage** ` → reporting (Stage 7)`

Novel-site evaluation cannot reuse the M1-S/M2-S protocol
([Stage 6](06_evaluation.md)), because evaluating a novel-site model against the annotation it trained
on is **circular**. M3 has its own evaluator built around three principles — *independent truth*,
*held-out chromosomes*, and *per-gene (within-gene) ranking* — with the novelty post-filter applied as
part of the metric. The methodology is [methods §6](../../meta_layer/methods/06_m3_novel_site_formulation.md#6-anti-circular-evaluation-the-core-contribution);
this stage is how to run it.

!!! abstract "Inputs → Outputs"
    **Reads:** a checkpoint (M3-S) or booster (M3-R), the D1/D2 truth sets, the annotation mask, and a
    per-gene `.npz` cache (recognizer) or the feature parquets (refiner).
    **Writes:** `m3_eval_metrics.json` + a comparison table (base vs each model on D1 / D1_hiconf / D2).

---

## The two evaluators

| Model | Script | How it scores |
|-------|--------|---------------|
| **M3-S** (recognizer) | `13_evaluate_m3_novel.py` | dense per-position inference → post-filter → per-gene precision@k |
| **M3-R** (refiner) | `15_evaluate_candidate_classifier.py` | rerank the candidate pool → same metric library |

Both report **per-gene precision@k / recall@k** (k = 5/10/20) over the truth-containing genes on the
held-out chromosomes (1, 3, 5, 7, 9), against three truth sets:

- **D1** — ENCODE long-read novel junctions (131,820 sites);
- **D1_hiconf** — the ≥ 2-biosample subset (53,956);
- **D2** — held-out disease anchors (171 novel).

The novelty **post-filter** (subtract annotated sites before ranking) is built into the metric, so
precision@k measures novelty recovery, not "is this a canonical site."

---

## M3-S — recognizer eval (three modes)

`13` separates the one GPU-dependent step (the feature cache) from the cheap local steps:

```bash
# 1. LOCAL — resolve the truth-containing test-gene universe (no pod)
python examples/meta_layer/13_evaluate_m3_novel.py --mode emit-universe

# 2. POD — build the shared per-gene .npz cache (bigWig; the only pod step)
python examples/meta_layer/13_evaluate_m3_novel.py --mode build-cache \
    --gene-list output/meta_layer/m3_eval_d1/eval_genes.txt \
    --cache-dir output/meta_layer/m3_eval_d1/gene_cache \
    --bigwig-cache /runpod-volume/bigwig_cache

# 3. LOCAL — score every model on the cache, report precision@k
python examples/meta_layer/13_evaluate_m3_novel.py --mode eval \
    --cache-dir output/meta_layer/m3_eval_d1/gene_cache
```

`--mode eval` scores **base / M1-S / M2-S / M3-v1 / M3-v1-mm0 / M3-v1.1** on the *same* candidate set, so
every number is meta-vs-base on identical genes. It **fails fast** if base scores are missing (they would
silently become a uniform prior and invalidate the eval) and loads each checkpoint via the config-type
dispatcher `load_meta_model` (not the `concat_fusion`-hardcoded path in `08`/`09`).

## M3-R — refiner eval

`15` reranks the candidate pool by base score vs M3-R and scores both through the **same** metric library
(`_m3_novel_eval.py`), by rasterizing per-candidate scores into the `[L,3]` array the recognizer path
uses (with `-inf` fill so the precision@k denominator stays candidate-native):

```bash
python examples/meta_layer/15_evaluate_candidate_classifier.py \
    --universe output/meta_layer/m3_eval_d1/eval_genes.parquet
```

---

## What the results say (Tier 0 / 1 / 2)

The evaluator was built as **Tier 0** of an improvement ladder; the rungs and their honest outcomes:

| Tier | Experiment | Verdict |
|------|-----------|---------|
| **0** | Build the anti-circular eval, score M3-v1 | **M3-v1 is the best novel ranker** (D1 P@5 0.335 vs base 0.277; D2 R@20 0.79 vs 0.51); multimodal adds little |
| **1** | Confirmed-only recognizer retrain | **No help** — `m3s_v1_1_confirmed` marginally worse; label noise wasn't the ceiling |
| **2** | M3-R candidate refiner | **Honest negative** — trains to AUC 0.90 but **ties base**; its edge is between-gene, not within-gene |

The full tables and the within-vs-between-gene decomposition that explains Tier 2 are in
[results/m3_novel.md](../../meta_layer/results/m3_novel.md).

!!! tip "The reusable evaluation idea"
    Report **per-gene precision@k against independent truth**, not genome-wide PR-AUC against the training
    annotation. And when a model's global AUC and its per-gene precision@k disagree, decompose into
    within-gene vs between-gene — that decomposition is what turned M3-R from a mystery into a precise,
    reusable lesson (genome-averaged multimodal tracks are locus-level, not position-level).

---

## Inspect the ranker interactively — the Novel Site Explorer

The aggregate precision@k numbers say M3 ranks well; the **Novel Site Explorer** lets you see it
per gene. Launch the Bio Lab UI and open a gene:

```bash
conda run -n agentic-spliceai python -m server.bio.app   # http://localhost:8005/novel/TPR
```

For a gene it shows M3's top-k candidate **unannotated** sites with the novelty post-filter applied,
each row carrying its M3 score, the base model's score and rank at the same position, the splice
dinucleotide, and **independent** evidence badges — ENCODE long-read support (D1, with biosample
count) and held-out disease anchors (D2, with mechanism). SpliceVault/`positives_pooled` is
deliberately *not* shown as evidence: it is M3's training pool, so it would be circular.

The serving universe is the same held-out set this stage evaluates (test chromosomes 1/3/5/7/9), so
every inspectable gene is one M3 never trained on. `TPR` is a good first look — both of its
SF3B1-cryptic anchors land in the top 5, at base scores of ~0.09–0.13.

Implementation: `server/bio/m3_inference.py`, sharing the ranking primitives in
[`splice_engine/eval/novel_site_ranking.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/src/agentic_spliceai/splice_engine/eval/novel_site_ranking.py)
with this stage's harness, so the UI and the eval rank identically by construction.

## Related

- Tissue-stratified evaluation (an M2-S-oriented protocol, applicable when tissue context matters):
  [`examples/meta_layer/16_evaluate_tissue_stratified.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/16_evaluate_tissue_stratified.py).
- Local dev results: [`examples/meta_layer/docs/M3/`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/M3/)
  (`m3_eval_D_results.md`, `m3r_candidate_refiner_results.md`).

---

→ **Reporting:** roll these into the [results pages](../../meta_layer/results/m3_novel.md) as in
[Stage 7 — Reporting](07_reporting.md).

# M3-R candidate refiner — results (Tier 2)

**Date:** 2026-07-27 · **Verdict:** honest negative — the multimodal candidate
refiner does **not** improve novel-site ranking over the base model on the
anti-circular test, and we now know precisely why.

Scripts: [`14_train_candidate_refiner.py`](../../14_train_candidate_refiner.py)
(Phase 1) · [`15_evaluate_candidate_classifier.py`](../../15_evaluate_candidate_classifier.py)
(Phase 2). Training-data prep: [`m3_training_data.md`](m3_training_data.md) §5.

## The reframe (recap)
Tier 0 found M3-v1 is the best novel-site ranker but its multimodal channels add
almost nothing genome-wide. M3-R reframes: the base model proposes candidates; an
XGBoost classifier reranks each **real cryptic vs artifact** from multimodal
evidence, trained with **base-score-matched hard negatives** so base can't be the
discriminator. Fully local.

## Phase 1 — training looked great
Held-out (SpliceAI test chr1/3/5/7/9): logistic base 0.811 → base+multimodal
**0.884 (+0.074)**; **XGBoost AUC 0.903, PR-AUC 0.73**; **56% of SHAP is
non-base** (genomic 22%, epigenetic 16%, conservation 12%). By that number the
reframe "worked."

## Phase 2 — the anti-circular test says otherwise
Rerank the per-gene candidate pool by base score vs M3-R, novelty post-filtered,
precision@k/recall@k vs independent long-read (D1) + disease (D2) truth. **They tie:**

| truth | metric | base | M3-R |
|---|---|---|---|
| D1 | P@5 / R@20 | 0.297 / 0.116 | 0.296 / 0.116 |
| D1_hiconf | P@5 / R@20 | 0.233 / 0.170 | 0.231 / 0.170 |
| D2 | P@5 / R@20 | 0.059 / 0.178 | 0.059 / 0.184 |

(base P@5 0.297 ≈ Tier 0's 0.277 → the harness is sound. Absolute recall is lower
than Tier 0 because the candidate pool is the parquet-sampled set, not dense.)

## Why — within-gene vs between-gene
M3-R is *not* copying base (Spearman 0.30; top-5 differs in 97% of genes) — it
reorders aggressively, but **orthogonally to truth**. Decomposing the AUC of
discriminating novel D1-truth in the post-filtered candidate pool:

| | base | M3-R |
|---|---|---|
| **global** (pooled across genes) | 0.585 | **0.675** |
| **within-gene** (mean per-gene) | 0.602 | 0.592 |

**M3-R's entire advantage is between-gene, not within-gene.** It scores
truth-rich loci higher overall (global +0.09), but *within* a gene it ranks
candidates no better than base (a hair worse). precision@k is a within-gene top-k
metric, so the between-gene gain is invisible to it.

The reason is mechanistic: the multimodal features M3-R leans on — epigenetic
(H3K36me3 = actively-transcribed exon), genomic (GC / gene position), conservation
— are **gene/context-level**, ~constant across a gene's candidate positions. They
say "this locus is the *kind of place* novel sites occur," not "*which base* is the
real site." The Phase-1 AUC of 0.90 was inflated by exactly this between-gene
signal (real sites sit in expressed/conserved genes; sampled artifacts often don't).
Base score, being position-specific, remains the better within-gene ranker.

This unifies every M3 multimodal result: Tier 0's +0.016 genome-wide, Tier 0's D2
"multimodal hurts" (cell-type mismatch), and Tier 2's train-0.90 / eval-tie — all
the same fact. **Genome-averaged multimodal tracks carry between-gene (locus) signal,
not within-gene (position) signal; novel-site discovery is a within-gene ranking
problem, so they don't help it.**

## Forward options (honest)
1. **Position-level features** — the real lever for within-gene ranking. Current
   tracks are tissue-averaged / gene-level; base-pair-resolution, splice-proximal
   features (local RBP motifs, RNA structure, delta-style local signals) is a
   features problem, not a framing one.
2. **Repurpose the between-gene signal** — M3-R's global AUC 0.675 is a legitimate
   **gene/locus triage** ("which genes harbor cryptic sites — inspect these first"),
   a useful discovery-pipeline capability even though it's not within-gene ranking.
3. **M3-v1 stays the within-gene deliverable** — best novel ranker; multimodal
   doesn't improve it. Ceiling documented.
4. **M4 (perturbation-conditional)** — directional, position-specific signal by
   construction; the frame where multimodal / mechanism actually moves the needle.

Raw metrics: `output/meta_layer/m3_eval_d1/m3r_eval_metrics.json` +
`output/meta_layer/m3r_candidate_refiner/metrics.json`.

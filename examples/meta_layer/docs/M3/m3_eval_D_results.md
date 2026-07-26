# M3 Phase D / Tier 0 — evaluation results

**Date:** 2026-07-26 · **Model under test:** `output/meta_layer/m3_v1` (3-class
novel-site recognizer, mm_channels=7) · **Harness:**
[`examples/meta_layer/13_evaluate_m3_novel.py`](../../13_evaluate_m3_novel.py)
(+ [`_m3_novel_eval.py`](../../_m3_novel_eval.py)).

## What this measures (and why)

The in-distribution validation number (donor PR-AUC 0.297 / acceptor 0.318) was
measured on a holdout of the *same* SpliceVault-dominated, ~50%-long-read-confirmed
label pool M3 trained on — it tells us training ran, not whether M3 finds novel
sites that are *independently* supported by biology. This eval replaces it with:

- **Use-case-shaped metric:** per-gene **precision@k / recall@k** (k = 5/10/20),
  not genome-wide PR-AUC. M3's real use is "for this locus, give me the top
  candidate unannotated sites to inspect."
- **Anti-circular truth:** eval universe = SpliceAI **test chromosomes 1/3/5/7/9**
  (held out for M1-S, M2-S, and M3-v1 alike). Truth =
  - **D1** = ENCODE long-read novel junctions (independent of the SpliceVault
    training signal); **D1_hiconf** = D1 with `n_biosamples ≥ 2`.
  - **D2** = held-out novel disease anchors (TDP-43 / SF3B1 / ENCODE-KD).
- **Novelty post-filter:** annotated sites are set-subtracted (4-tuple anti-join
  against GENCODE ∪ RefSeq) *before* ranking — so precision@k measures novelty,
  not "is this a canonical site."
- **Five models on identical candidates:** `base`/OpenSpliceAI (the base-score
  channel — sequence-only), `M1-S`, `M2-S`, `M3-v1`, and `M3-v1-mm0` (M3 with all
  multimodal channels zeroed → isolates the multimodal contribution).

Universe: **4,956 truth-containing test-chrom genes**, 0 skipped.
D1 = 131,820 truth sites over 9,394 gene×type units; D1_hiconf = 53,956 / 8,036;
D2 = 171 / 163.

## Results (per-gene macro, donor+acceptor combined)

### D1 — ENCODE long-read novel (all)
| model | P@5 | R@5 | P@10 | R@10 | P@20 | R@20 |
|---|---|---|---|---|---|---|
| base | 0.277 | 0.160 | 0.217 | 0.232 | 0.160 | 0.315 |
| M1-S | 0.228 | 0.133 | 0.175 | 0.189 | 0.127 | 0.256 |
| M2-S | 0.258 | 0.150 | 0.208 | 0.221 | 0.157 | 0.308 |
| **M3-v1** | **0.335** | **0.176** | **0.269** | **0.263** | **0.202** | **0.367** |
| M3-v1-mm0 | 0.319 | 0.171 | 0.255 | 0.253 | 0.191 | 0.353 |

### D1_hiconf — long-read novel, ≥2 biosamples
| model | P@5 | R@5 | P@10 | R@10 | P@20 | R@20 |
|---|---|---|---|---|---|---|
| base | 0.208 | 0.222 | 0.156 | 0.312 | 0.110 | 0.413 |
| M1-S | 0.164 | 0.177 | 0.122 | 0.250 | 0.085 | 0.329 |
| M2-S | 0.196 | 0.210 | 0.151 | 0.303 | 0.108 | 0.407 |
| **M3-v1** | **0.254** | **0.247** | **0.194** | **0.358** | **0.139** | **0.485** |
| M3-v1-mm0 | 0.241 | 0.238 | 0.185 | 0.345 | 0.133 | 0.469 |

### D2 — held-out novel disease anchors
| model | P@5 | R@5 | P@10 | R@10 | P@20 | R@20 |
|---|---|---|---|---|---|---|
| base | 0.061 | 0.294 | 0.041 | 0.393 | 0.027 | 0.506 |
| M1-S | 0.034 | 0.160 | 0.027 | 0.255 | 0.017 | 0.319 |
| M2-S | 0.054 | 0.252 | 0.038 | 0.359 | 0.025 | 0.482 |
| M3-v1 | 0.108 | 0.515 | 0.072 | 0.693 | 0.041 | 0.791 |
| **M3-v1-mm0** | **0.121** | **0.583** | 0.073 | **0.702** | **0.044** | **0.844** |

## Findings

1. **M3-v1 is the best novel-site ranker** — it beats `base`, `M1-S`, and `M2-S`
   on the anti-circular D1/D1_hiconf across *every* k, on both precision and
   recall, and on both donor and acceptor separately. It is the **only** meta
   model that improves over the raw base model on novel sites. (M1-S and M2-S
   *under*-perform base here — sensible: they were tuned toward *annotated* sites
   and mildly suppress novel ones.) So M3-v1 is a **usable ranker, not broken**:
   its top-5 novel candidates per gene contain ~1.7 long-read-confirmed sites, and
   it recovers ~49% of high-confidence novel sites within the top-20/gene.

2. **Yes, M3 adds value over the sequence-only base — the answer to the framing
   question — but the win is modest in this frame** (D1 P@5 0.335 vs 0.277;
   R@20 0.367 vs 0.315). Not a solved problem; a meaningful improvement.

3. **The multimodal channels contribute only a small slice of that edge.**
   M3-v1 vs M3-v1-mm0 (multimodal zeroed): D1 P@5 0.335 vs 0.319 (**+0.016**).
   Most of M3's advantage over base (mm0 already at 0.319 vs base 0.277) comes
   from the **learned meta head's recalibration** of base + sequence, not the
   conservation/epigenetic/RBP evidence. In the genome-scan frame the channels
   are underused — exactly the diagnostic the design predicted.

4. **D2 is the standout "recovers known biology" signal:** M3-v1 recovers **79%**
   of held-out novel disease cryptic sites in the top-20/gene (R@20 0.791) vs
   base 0.506; R@5 0.515 vs 0.294. **But** M3-v1-mm0 does *slightly better* on D2
   (R@20 0.844) — the multimodal tracks (K562/HepG2/neuronal cell types) appear
   **cell-type-mismatched** for these disease contexts and add mild noise. This
   argues for tissue/cell-type-matched multimodal features, or applying the
   channels surgically at candidate sites rather than genome-wide.

## Decision → Tier 2 (with Tier 1 folded in)

Per the runbook gate, this is the **precise-top / usable-ranker** outcome, not
"low-precision-everywhere / broken." So the next step is **Tier 2 (candidate
refinement)** — reframe M3 as "is this base-proposed candidate a real cryptic
site or an artifact," the frame where the multimodal evidence should become
decisive (findings 3 & 4 show it is wasted in the genome-scan frame). **Tier 1**
(retrain on the ~52–77K long-read-confirmed positives only, dropping the
unvalidated SpliceVault half) is the cheaper parallel experiment and folds in
naturally. Pure de-novo genome scanning is unlikely to clear the bar for a
drug-discovery pipeline regardless of labels/architecture.

Raw metrics: `output/meta_layer/m3_eval_d1/m3_eval_metrics.json`.

# M3 — Novel Splice Sites

**Task.** Find **novel** splice sites — positions absent from *every* annotation (MANE, Ensembl,
GENCODE, RefSeq) yet supported by real evidence (junction reads, long-read isoforms, disease data).
This is the discovery frontier and the hardest of the four tasks. Because a novel site has no
annotation label, **junction support is used as the training label and removed from the input
channels** (mm_channels = 7) to avoid target leakage.

**Bottom line — a three-part milestone:**

- **M3-S (M3-v1)** is the **best general novel-site ranker** in the system — the only meta model that
  beats the raw base model on independent novel-site truth.
- **M3-anchor** folds curated disease anchors into training at a governed budget. It **improves held-out
  SF3B1 disease-cryptic ranking (D2 R@20 0.79 → 0.86) with no cost to general recall (D1)** — the first
  meta result to move the disease-cryptic needle, though cross-mechanism transfer to the ALS sites stays
  weak.
- **M3-R**, the candidate refiner, is an **honest negative**: it trains to a strong AUC (0.90) but
  **ties the base model** on the anti-circular test. Understanding *why* produced the most useful
  finding of the M3 line — genome-averaged multimodal evidence carries **locus-level, not
  position-level**, signal.

---

## Evaluating novel-site discovery honestly

Evaluating a novel-site model against the annotation it trained on is circular. In-distribution, M3-v1's
validation PR-AUC is only 0.297 (donor) / 0.318 (acceptor) — which "tells us training ran, not whether
M3 finds novel sites." So M3 is scored **anti-circularly**, on independent truth sets, with the
annotated/training sites subtracted out:

- **D1** — ENCODE long-read novel junctions (131,820 sites across 9,394 gene×type units).
- **D1_hiconf** — the subset seen in ≥ 2 biosamples (53,956 sites).
- **D2** — held-out disease anchors (171 sites). **In practice an SF3B1 benchmark**: 171 SF3B1
  anchors, 2 ENCODE-KD, 0 TDP-43, and 171 of the 173 are acceptors. Every SF3B1 anchor sits within
  50 nt of an annotated site of the same type (median 17 nt), because SF3B1 mutations shift
  branch-point selection rather than activating a distant cryptic exon.

Universe: 4,956 truth-containing genes on the held-out chromosomes (1, 3, 5, 7, 9). The metric is
**per-gene precision@k / recall@k** — a *within-gene* ranking metric (can this model rank the real novel
site above the decoys *in the same gene?*). That within-gene framing is central to the M3-R result below.

---

## M3-S (M3-v1) — the novel-site ranker

A 3-class sequence-CNN recognizer (the M2-S backbone with junction dropped) that ranks candidate novel
sites per gene. On the anti-circular D1 truth, it is the only meta model that improves on base:

!!! info "What **M3-v1** names"
    Canonical ID **`m3s.concat_fusion.cleanannot`** — variant M3-S, `concat_fusion` architecture,
    `cleanannot` corpus. "v1" is the **training run**, not an architecture or a corpus generation;
    it is the first M3 fit, and `M3-v1.1` below is the confirmed-only retrain of the same three
    axes. Checkpoint: `output/meta_layer/m3_v1/`. See the
    [naming convention](../methods/naming_convention.md).

| Model | P@5 | R@5 | P@20 | R@20 |
|-------|-----|-----|------|------|
| Base | 0.277 | 0.160 | 0.160 | 0.315 |
| M1-S | 0.228 | 0.133 | 0.127 | 0.256 |
| M2-S | 0.258 | 0.150 | 0.157 | 0.308 |
| **M3-v1** | **0.335** | **0.176** | **0.202** | **0.367** |
| M3-v1 (multimodal zeroed) | 0.319 | 0.171 | 0.191 | 0.353 |

- On **D1_hiconf**, M3-v1 reaches P@5 0.254 and **R@20 0.485** — recovering ~49% of high-confidence
  novel sites in the top-20 per gene.
- On **D2** disease anchors, M3-v1 **R@20 0.791** (79%) vs base 0.506; R@5 0.515 vs 0.294. Read this
  as an **SF3B1** result (see the D2 note above), not as a general disease-cryptic claim: the model
  is ranking cryptic acceptors a median 17 nt from a canonical one. Performance on
  TDP-43-style cryptic exons, which sit deep inside introns, is **not measured by D2** and remains
  open.

Note two things. First, **M1-S and M2-S actually underperform the base model here** — canonical/alternative
refiners are the wrong tool for novel sites. Second, the **multimodal contribution is small**: M3-v1 beats
its own multimodal-zeroed variant by only +0.016 (P@5 0.335 vs 0.319), and on D2 the zeroed variant is
*slightly better* (R@20 0.844) — cell-type-mismatched external tracks add mild noise. **Most of M3-v1's
advantage comes from the learned meta head's recalibration, not from the multimodal evidence.** Hold
that thought.

**Conclusion.** M3-v1 is the best novel-site ranker and the current within-gene deliverable for the
discovery task.

!!! tip "See it per gene — the Novel Site Explorer"
    These are aggregate numbers; the Bio Lab UI renders the ranking itself. Run
    `python -m server.bio.app` and open `/novel/TPR` for a gene's top-k candidate unannotated sites,
    each with the base model's score and rank alongside M3's, the splice dinucleotide, and independent
    evidence badges (ENCODE long-read support, held-out disease anchors). Serving is restricted to the
    same held-out chromosomes evaluated here, so every inspectable gene is one M3 never trained on.

### Tier 1 — does cleaner labels help? (no)

M3-v1 trains on all 154,113 pooled positives with long-read-confirmed ones up-weighted 2×. Only ~50%
are long-read-confirmed, so the original diagnosis blamed that unconfirmed half as "the ceiling — label
noise you can't learn a clean boundary through." **Tier 1 tested it directly:** retrain the recognizer on
the **77,879 confirmed positives only** (`07 --confirmed-only`, dropping 76,234), same architecture and
schedule, then re-run the identical anti-circular eval.

| Model | D1 P@5 | D1 R@20 | D1_hiconf P@5 | D2 R@20 |
|-------|--------|---------|---------------|---------|
| **M3-v1** (all pos, confirmed 2×) | **0.335** | **0.367** | **0.254** | **0.791** |
| M3-v1.1 (confirmed-only) | 0.327 | 0.343 | 0.249 | 0.779 |

**M3-v1.1 is marginally *worse* everywhere — the hypothesis is refuted.** Dropping the unconfirmed
positives lost signal (more data won) rather than removing noise; the unconfirmed SpliceVault sites are
not pure artifacts. Label noise was not the ceiling — which, together with M3-R below, points the
remaining leverage away from labels and framing and toward **position-level features**.

---

## M3-anchor — folding disease anchors in (the disease-cryptic win)

Tier 1 and M3-R point the leverage for *general* novel sites away from labels. This section shows a
targeted exception for the *disease-specific* metric. The question: does adding the curated **disease
anchors** as training positives help? M3-v1 lacks them entirely (SpliceVault/GTEx come from healthy
tissue), and until recently they were worse than absent: they fell through to class-2 "neither",
teaching the recognizer that real SF3B1 / ENCODE-KD / TDP-43 cryptic sites are *not* splice sites.
**M3-anchor** folds them in properly (checkpoint `output/meta_layer/m3s_anchorpos`; see the
[training workflow](../../workflows/meta_layer/10_training_m3.md)):

- **Positivize** the SF3B1 and ENCODE-KD anchors (`07 --disease-anchors positivize`), **mask** the
  TDP-43 (STMN2/UNC13A) anchors so they stay honest held-out probes.
- **Governed by a budget, not a multiplier.** The anchors occupy a fixed **5% of positive-loss mass**
  (`--anchor-budget-frac 0.05`), which derives a per-anchor weight of 21.8×. A flat 5× would have been
  only ~1.2% of the signal, too quiet to register; the budget framing is what let the anchors count.
- **Anti-circular by chromosome:** only train-chromosome genes are cached, so the 171 held-out SF3B1
  anchors (= D2, on chr 1/3/5/7/9) are never trained on.

### It improves disease-cryptic ranking without costing general recall

**D1 (general novel sites) holds** — folding anchors in did not trade away breadth:

| Model | D1 P@5 | D1 P@20 | D1 R@20 |
|-------|--------|---------|---------|
| base | 0.277 | 0.160 | 0.315 |
| M3-v1 | 0.335 | 0.202 | 0.367 |
| M3-anchor | 0.333 | 0.200 | 0.364 |

**D2 (held-out SF3B1 disease anchors) improves across every metric:**

| Model | D2 P@5 | D2 R@5 | D2 R@10 | D2 R@20 |
|-------|--------|--------|---------|---------|
| base | 0.061 | 0.294 | 0.393 | 0.506 |
| M3-v1 | 0.108 | 0.515 | 0.693 | 0.791 |
| **M3-anchor** | **0.120** | **0.571** | **0.755** | **0.859** |

Held-out SF3B1 R@20 rises **0.791 → 0.859 (+6.8 pts)** and R@5 **0.515 → 0.571**, on a disjoint
chromosome set from the training anchors. The recognizer genuinely learned a transferable SF3B1
cryptic-acceptor pattern.

### Cross-mechanism transfer to the ALS sites is mixed

The TDP-43 anchors were masked (never trained), so STMN2/UNC13A test whether SF3B1-learning transfers to
a *different* mechanism. Within-gene rank of each cryptic site among its gene's novel candidates:

| Cryptic site | base | M3-v1 | M3-anchor |
|--------------|------|-------|-----------|
| STMN2 acceptor | #1 | #4 | **#2** |
| UNC13A donor | #143 | #260 | **#187** |
| UNC13A acceptor | #1312 | #660 | #872 |

Two of three improve, one regresses. SF3B1-learning helps the ALS sites only partially and unreliably,
consistent with SF3B1 (branch-point 3′SS shift) and TDP-43 (deep-intronic de-repression) being different
biology.

### Verdict

A real, honest positive, *within-mechanism*. The budgeted anchor fold lifts held-out SF3B1 ranking with
no D1 cost, validating both the disease-anchor approach and the budget-governance framing (the flat-5×
version would have been inaudible). Cross-mechanism transfer to the weak, perturbation-gated regime
(UNC13A) is weak, which is [M4](#m4-perturbation-induced-in-progress)'s job rather than SF3B1 transfer.
**Not yet promoted:** a single chromosome split, so a chromosome-fold CV would tighten the estimate
before M3-anchor replaces M3-v1.

---

## M3-R — the candidate-refiner milestone

M3-R reframes discovery as **candidate refinement**: let the base model propose candidate sites, then
train an XGBoost classifier to rerank each candidate as *real cryptic site vs artifact* from the
multimodal evidence. The key design choice is **base-score-matched hard negatives** — negatives are
sampled to have the same base-score distribution as positives, so the base score *cannot* be the
discriminator and the multimodal channels are forced to earn their keep.
(Scripts:
[`14_train_candidate_refiner.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/14_train_candidate_refiner.py) /
[`15_evaluate_candidate_classifier.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/15_evaluate_candidate_classifier.py).)

### Phase 1 — training looked strong

On the held-out test chromosomes (1, 3, 5, 7, 9; 52,320 candidates, 13,105 real, 88 features):

- Logistic probe: base-only AUC 0.811 → base + multimodal **0.884 (+0.074)**.
- **XGBoost: AUC 0.903, PR-AUC 0.73.**
- **56% of the SHAP importance is non-base** (genomic 22%, epigenetic 16%, conservation 12%) — exactly
  what the base-matched-negative design was meant to produce.

By every training-time signal, the reframe worked: the multimodal channels are decisive.

### Phase 2 — but the anti-circular test ties base

Reranking the same candidate pool and scoring it against the independent D1/D2 truth with per-gene
precision@k / recall@k, M3-R **matches the base model to within ±0.002 everywhere**:

| Truth | Metric | Base | M3-R |
|-------|--------|------|------|
| D1 | P@5 / R@20 | 0.297 / 0.116 | 0.296 / 0.116 |
| D1_hiconf | P@5 / R@20 | 0.233 / 0.170 | 0.231 / 0.170 |
| D2 | P@5 / R@20 | 0.059 / 0.178 | 0.059 / 0.184 |

### Why — between-gene vs within-gene

Decomposing the AUC resolves the contradiction:

| AUC | Base | M3-R |
|-----|------|------|
| Global (pooled across all candidates) | 0.585 | **0.675** (+0.09) |
| Within-gene (mean of per-gene AUCs) | 0.602 | 0.592 (−0.010) |

**M3-R's entire advantage is between-gene, not within-gene.** It has learned to tell "this is the kind
of *locus* where cryptic sites occur" — but not "this is the *base* within the locus that is the real
site." The per-gene precision@k metric only rewards the latter, so the gain is invisible there
(Spearman vs base 0.30; the top-5 ranking differs in 97% of genes, yet performs the same).

**Root cause:** the external multimodal tracks (conservation, epigenetic, chromatin, RBP) are
**gene/locus-level** — approximately constant across the tens-of-bp span of a gene's candidate
positions. They signal *what kind of locus* this is, which is between-gene information; they cannot
discriminate *which base* is the site, which is what discovery needs. The impressive Phase-1 AUC of 0.90
was inflated by exactly this between-gene signal.

### Why this is the key M3 finding

This single result **unifies every earlier "multimodal barely helps M3" observation**: M3-v1's +0.016
genome-wide multimodal gain, the "multimodal slightly hurts on D2" effect, and now Phase-2's
train-0.90 / eval-tie. They are all the same phenomenon — genome-averaged tracks carry locus-level, not
position-level, information.

**Verdict: honest negative.** M3-R is not promoted. But it is not wasted:

- The global AUC 0.675 is **repurposable as gene/locus-level triage** — ranking *which genes* to search,
  not which base.
- **M3-v1 remains the within-gene deliverable.**
- The path forward is **position-level features** (evidence that varies base-to-base, e.g. foundation-model
  scalars or local junction reads), and the perturbation-conditional [M4](#m4-perturbation-induced-in-progress).

---

## M4 — perturbation-induced (in progress)

M4 targets splice sites induced by perturbation (e.g. TDP-43 loss in ALS). Current status: M2-S already
*detects* the UNC13A cryptic donor (score 0.517), but the **de-repression logic is not yet learned** —
that requires perturbation-paired training labels. No evaluated results yet; tracked as active research.

---

## Takeaways

- **M3-v1 is the best general novel-site ranker** — the only meta model to beat raw base on independent
  novel truth (D1 P@5 0.335 vs 0.277; D2 R@20 0.79 vs 0.51).
- **M3-anchor moves the disease-cryptic metric:** folding SF3B1/KD anchors in at a governed 5% budget
  lifts held-out SF3B1 D2 R@20 to 0.86 with no D1 regression — but the win is within-mechanism, and
  transfer to the perturbation-gated ALS regime (UNC13A) stays weak (that is M4's job). The budget
  framing was load-bearing: a flat 5× (~1.2% of positive mass) would have been inaudible.
- **M3-R is an instructive negative:** strong training AUC (0.90, 56% non-base SHAP) but no anti-circular
  gain, because its edge is between-gene while discovery is within-gene.
- The reusable lesson for the whole meta layer: **genome-averaged multimodal tracks are locus-level
  evidence.** They lift *canonical* and *alternative* recognition (where the question is partly "what kind
  of locus"), but *novel-site discovery* needs position-level signal they don't provide.

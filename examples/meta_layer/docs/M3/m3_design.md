# M3 v1 — Design Synthesis

The design for **M3 v1**, the unconditional novel splice site discoverer.
This consolidates the scope decision, the label strategy, the architecture
position, and how the disease-context idea is incorporated. It is the public
companion to the prerequisites ([`m3_prerequisites.md`](m3_prerequisites.md))
and the experimental audit ([`label_audit_A1.md`](label_audit_A1.md)).

> **Status (reconciled 2026-07-30).** This note was written **2026-05-26**, before
> M3 was trained (05-30) or evaluated (07-26/27). It has been reconciled against
> what shipped. Sections carrying a **✅ / ⚠️ Outcome** callout were tested; the
> callout is what the experiment returned, and where it contradicts the original
> design reasoning the callout wins. Two predictions were wrong and are corrected
> inline: the negative-sampling scheme in §5 was never used, and the "label
> quality is the bottleneck" claim in §3 was refuted. §9 records the final
> outcomes; the current numbers live in
> [`m3_eval_D_results.md`](m3_eval_D_results.md) and
> [`../../../../docs/meta_layer/results/m3_novel.md`](../../../../docs/meta_layer/results/m3_novel.md).

---

## 1. Why unconditional first

M3's job is to surface splice sites that **no annotation knows about** but
that junction evidence supports. There are two ways to frame the model:

1. **Unconditional** — "does this position have splice-site potential that
   the annotation missed?" Produces a genome-wide candidate catalog.
2. **Disease-conditional** — "given perturbation state X (TDP-43 loss,
   SF3B1 mutation, …), does this position become an active splice site?"
   Produces per-condition predictions.

We build **unconditional first**, for six reasons:

| | Unconditional M3 | Disease-conditional (M3-D) |
|---|---|---|
| Serves Phase 8 isoform discovery directly | Yes (catalog generator) | No (only trained contexts) |
| Generalises to a *new* disease without retraining | Yes | No |
| Positives available at train time | Pooled across all sources (tens of thousands) | Per-disease (TDP-43 ≈ 150) |
| Risk of a "disease-ID lookup" shortcut | None | Real |
| Thesis it tests | "we can find novel sites from sequence + multimodal evidence" | "we can predict perturbation effects given the perturbation" (this is the **M4** thesis) |
| Upfront data engineering | GTEx (already on disk) + small published catalogs | Multi-cohort recount3 ELT |

The disease-conditional framing is genuinely valuable — but it is the
[M4](../M4/) thesis (regulator → splicing map, perturbation as input), not
M3's. Folding it into M3 v1 would collapse novel-discovery into
perturbation-prediction and discard the unconditional catalog that Phase 8
and the agentic layer actually consume.

**The disease catalogs are still used** — see §3 — just as *pooled positive
labels*, not as a conditioning input.

> **Note (2026-07-30).** The "upfront data engineering" row above has since been
> paid down on the M4 side: a perturbation-paired ΔPSI corpus now exists
> (`data/mane/GRCh38/m4_labels/kd_dpsi.parquet` — 3.17M signed per-site ΔPSI rows
> over 185 RBPs from ENCODE shRNA-KD rMATS). That does not change the
> unconditional-first decision — the two models answer different questions — but
> the conditional sibling is no longer data-blocked. See [`../M4/`](../M4/).

---

## 2. Junctions as label, not input

M1 and M2 take junction support as an **input** feature (the `junction`
modality, 12 columns). M3 cannot: junction evidence *is* the supervision
signal for "this is a real splice site the annotation missed." Using it as
an input would leak the target.

So in M3:
- The `junction` modality is **removed from the input channels**.
- Junction support (depth, tissue breadth, the GTEx-novel survivor set)
  becomes the **positive label**.
- The remaining modalities (sequence, base scores, conservation, epigenetic,
  RBP, chromatin accessibility) stay as inputs — they are what the model uses
  to *predict* novelty from context.

---

## 3. Label strategy — pooled, not conditioned

Positives come from three pools, each tagged with provenance so they can be
ablated later:

| Pool | Source | Provenance tag |
|---|---|---|
| GTEx-novel survivors | Cross-annotation audit (A1): junction-supported, absent from Ensembl ∪ GENCODE ∪ RefSeq-curated, passing depth/breadth filters | `gtex_novel` |
| Tier-1 disease catalogs | TDP-43 cryptic exons (Brown 2022), SF3B1 cryptic 3′SS, U2AF1 cassette exons, SMN2 paradigm | `tdp43_cryptic`, `sf3b1_cryptic`, `u2af1_cryptic`, `smn2_paradigm` |
| (deferred) Tier-2 cohorts | TCGA / AMP-AD / AMP-PD via recount3 — disease-Δ labels | not in v1 |

**Corrected pool (2026-05-24).** The pooled positive set is **154,113 novel
splice sites**, dominated by SpliceVault:

| Source | Sites | Notes |
|---|---:|---|
| SpliceVault-novel | ~153,855 | cryptic donors/acceptors, GRCh38-native, 100% GT/AG |
| GTEx-novel | 748 | genuinely novel after the annotation bug fix (see below) |
| both | 40 | |

> The earlier A1 figure of "65,163 GTEx-novel sites" was a bug artifact — the
> annotation `splice_sites_enhanced.tsv` files mis-placed minus-strand splice
> sites, so 52,710 minus-strand *annotated* sites leaked into the GTEx "novel"
> pool. Fixed at source (`extract_splice_sites_from_exons`) and regenerated.
> See [`label_audit_A1.md`](label_audit_A1.md) correction note.

The held-out disease anchors (TDP-43, SF3B1, ENCODE-KD) total **6,351 sites**
(563 novel), reserved for the Phase D2 generalization test, anti-joined out of
the training pool. The bottleneck for M3 is therefore **not** finding
candidates — it is *label quality* (separating real cryptic biology from
alignment artifacts and low-frequency noise), addressed by the Phase B filter
stack and the Phase D long-read truth set.

> **⚠️ Outcome — the label-quality claim was tested and refuted.** Phase B3 showed
> only ~50% of the pooled positives are long-read-confirmed, which made "the
> unconfirmed half is the ceiling" the natural diagnosis. **Tier 1 tested it
> directly:** retrain the recognizer on the 77,879 confirmed positives only
> (`07 --confirmed-only` → `m3s_v1_1_confirmed`), same architecture and schedule.
> The result is **marginally worse everywhere** (D1 P@5 0.327 vs 0.335; D1 R@20
> 0.343 vs 0.367; D2 R@20 0.779 vs 0.791). Dropping the unconfirmed positives
> lost signal rather than removing noise — more data won, and the unconfirmed
> SpliceVault sites are not pure artifacts. **Label quality was not the
> bottleneck.** See §9.

---

## 4. Architecture — the lever is not architecture

The original M3 sketch (and the early review) assumed novel sites need a
**wider receptive field** than M1/M2 — that cryptic sites depend on context
the ~400 bp RF misses, motivating dilations out to RF ≈ 2.6 kb and possibly
attention.

**A controlled M2-S ablation (2026-05-17 → 05-19) falsified the architecture
lever:**

- Widening the receptive field did **not** help (accuracy −0.028).
- Global self-attention was *catastrophic* (val PR-AUC 0.665 vs 0.833) —
  averaging the rare positive signal away over a ~6 kb window where >99% of
  positions are non-splice.
- Restricting attention to a **local window** recovered the baseline
  (0.802 vs 0.807) but did **not exceed** it.

Conclusion for the meta layer generally, M3 included: **attention and wider
RF are at best neutral and at worst harmful for sparse splice-site
prediction. The lever is data and labels.** (Full post-mortem in the M2-S
dev notes.)

This reshapes M3 v1:

- **v1 keeps the M2-S backbone essentially as-is** — the `concat_fusion`
  architecture (3-stream dilated CNN, cat-fusion, RF ≈ 400 bp, single 3-class
  head). Junction modality dropped from inputs (§2).
- The earlier "widen dilations to RF ≈ 2.6 kb" change is **demoted to an
  optional follow-up experiment**, run *only if* a label-side baseline plateaus
  and diagnostics implicate context length — not as a default.
- Engineering effort goes to **label assembly and negative sampling**
  (Phase B), where the M2-S work says the gains actually live.

> **✅ Outcome — the architecture call was right; the "labels" half was not.**
> M3-v1 shipped as the M2-S `concat_fusion` backbone *verbatim* apart from the input change —
> `MetaSpliceConfig(variant='M3-S', hidden_dim=32, seq_dilations=[1,1,1,1,4,4,4,4],
> mm_channels=7)`. The wider-RF experiment was never needed and remains unrun.
> But the second half of the conclusion — "the lever is data and labels" — was
> **half-refuted**: Tier 1 (§3) showed cleaner labels don't help, and Tier 2 (§9)
> showed reframing the task doesn't either. The remaining lever is neither
> architecture nor labels but **position-level features** — see §9.
>
> **Naming note.** The "v1.1" label proposed here for the wider-RF experiment was
> subsequently used for something else: `m3s_v1_1_confirmed` is the Tier 1
> *confirmed-only label* retrain at **identical** architecture. If the wider-RF
> experiment is ever run, give it a distinct tag.

---

## 5. Negative sampling — settled as "recognizer + post-filter" (2026-05-26)

**Decision:** M3 is a splice-site **recognizer**; novelty is applied at
inference by exact set-subtraction against annotation, NOT learned. This
resolves a flaw in the original "decoy" plan: a *novel* cryptic donor and an
*annotated* canonical donor are **sequence-identical**, so labeling annotated
donors as negatives while novel donors are positives creates contradictory
labels at identical motifs. Instead:

- **Positives** = the novel pooled sites (donor/acceptor) — `positives_pooled.parquet`.
- **Negatives** = true NON-sites (`negatives.parquet`, B2):
  - **hard** (154K): positions carrying a canonical GT/AG dinucleotide but that
    are neither annotated nor novel — forces the model past the bare
    dinucleotide (100% canonical by construction).
  - **easy** (154K): random gene-body positions without a canonical dinucleotide.
- **Annotated sites** = **MASKED** (ignore-index in the loss) via
  `annotation_mask.parquet` (826K sites) — neither rewarded nor penalized,
  avoiding the contradiction. (The earlier "decoy = annotated as negative" idea
  is dropped.)

Label set: **154,113 positives : 308,000 negatives (≈1:2)**, 826K masked. The
hard:easy ratio (here 1:1) is the tunable knob. Built by
`examples/data_preparation/m3/09_build_negatives.py`.

> **⚠️ Outcome — the recognizer/post-filter framing shipped; the sampled negative
> set did not.** The framing above is correct and is exactly what `--mode m3`
> implements (positives → donor/acceptor, annotated → `255` ignore-index,
> novelty applied as set subtraction at inference). What did *not* survive is the
> **sampled 1:2 label set**. M3-S trains **per gene window, not per site**: a
> single forward pass emits a dense 3-class array over the whole window
> (`0=donor, 1=acceptor, 2=neither, 255=ignore`), so *every* non-positive,
> non-masked position is already class 2. A per-site formulation would waste
> ~5000× the gradient signal on this dilated CNN.
>
> Consequently **`negatives.parquet` is unused by the recognizer** — training runs
> on the full gene split like M1/M2, and
> [`07_train_sequence_model.py`](../../07_train_sequence_model.py) prints
> `negatives.parquet unused` at startup. The "1:2 ratio" and the "hard:easy
> tunable knob" describe a knob that does not exist; the effective positive:negative
> ratio is whatever the gene windows contain. `09_build_negatives.py` remains
> useful for its **`annotation_mask.parquet`** output (which *is* load-bearing) and
> as an analysis artifact.
>
> The base-score-matched negatives that *were* decisive belong to **M3-R**
> (`candidate_labels.parquet`), a different formulation — §9 and
> [`m3_training_data.md`](m3_training_data.md) §5.

---

## 6. Evaluation — avoid circularity

Evaluating M3 on "sites not in Ensembl" alone is circular (that is how the
positives were defined). The headline evaluation uses an **independent truth
set**:

- **ENCODE4 long-read RNA-seq** (PacBio Iso-Seq + ONT): novel junctions
  confirmed by reads spanning a full transcript. This is the positive-truth
  set for the headline PR-AUC.
- **Per-disease held-out generalisation:** train with one Tier-1 catalog
  held out, measure recall on it. Tests whether M3 learned transferable
  cryptic-site grammar vs memorised per-disease coordinates.
- **Calibration:** reliability diagram before any calibration claim.

Win condition: M3 v1 beats the M2-S baseline on the long-read truth set by a
margin large enough to justify a separate model (target ≈ 5 PR-AUC points;
revisit once baselines are measured).

> **✅ Outcome — anti-circularity held; the metric and the protocol both changed.**
> The independent-truth principle is exactly what shipped, and it is the most
> valuable idea in this note. Three specifics differ from the sketch above:
>
> - **PR-AUC was abandoned as the headline.** M3's in-distribution validation
>   PR-AUC is 0.297 (donor) / 0.318 (acceptor) — measured on a holdout of the
>   *same* SpliceVault-dominated pool it trained on, so it reports that training
>   converged, not that M3 finds novel sites. The shipped metric is **per-gene
>   precision@k / recall@k** (k = 5/10/20) with the novelty post-filter applied
>   before ranking. It is use-case-shaped ("for this locus, give me the top
>   candidate unannotated sites") and, critically, it is a **within-gene** metric —
>   which is what makes the Tier 2 result in §9 legible.
> - **Generalisation is tested wholesale, not leave-one-catalog-out.** All 6,351
>   disease anchors (563 novel) are held out as **D2** and anti-joined from
>   training; anti-circularity comes from the SpliceAI **chromosome split**
>   (test = chr 1/3/5/7/9). Truth sets are **D1** (131,820 ENCODE long-read novel
>   junctions), **D1_hiconf** (the ≥2-biosample subset, 53,956), and D2.
> - **Calibration was never run** — and is not needed for a ranker scored by
>   precision@k. No calibration claim has been made.
>
> The win condition was met on the substituted metric: **D1 P@5 0.335 vs base
> 0.277 and M2-S 0.258; D2 R@20 0.791 vs base 0.506.** M3-v1 is the only meta model
> that beats raw base on independent novel-site truth — M1-S and M2-S both come in
> *below* base, confirming that a separate model was justified. Harness:
> [`13_evaluate_m3_novel.py`](../../13_evaluate_m3_novel.py).

---

## 7. What M3 v1 is not

- **Not** a truth-finder. Its output is a candidate list; downstream
  consumers (Phase 8, the agentic layer) apply structural / proteomic /
  cross-cohort filters.
- **Not** disease-conditional. That is M3-D / [M4](../M4/).
- **Not** tissue-conditional in v1. Tissue-specific novel sites will be
  under-ranked by the 54-tissue-averaged GTEx labels; tissue conditioning is
  a v2 consideration.

---

## 8. Phase status

All design-time phases are complete. (This section previously named Phase B1 as
"the next step" and quoted the pre-correction figure of 65,163 GTEx survivors —
a number §3 of this same document corrects to **748**.)

| Phase | What | Status |
|---|---|---|
| A1 | Cross-annotation audit | **Done** (corrected 05-24) — [`label_audit_A1.md`](label_audit_A1.md) |
| B1 | Positive-pool assembly | **Done** — 154,113 pooled positives + 6,351 held-out anchors |
| B2 | Negatives + annotation mask | **Done** — mask is load-bearing; sampled negatives unused (§5) |
| B3 | ENCODE4 long-read truth set | **Done** (05-28) — 681,809 novel D1 sites; `longread_confirmed` flag on the positive pool |
| C | Architecture + training | **Done** (05-30) — `output/meta_layer/m3_v1` |
| D | Anti-circular evaluation | **Done** (07-26) — [`m3_eval_D_results.md`](m3_eval_D_results.md) |

---

## 9. Outcomes — the improvement ladder (Tier 0 / 1 / 2)

Phase D was built as **Tier 0** of a ladder; each rung tested one hypothesis about
what limits novel-site discovery. Both later rungs came back negative, and
together they are more informative than either would be alone.

| Tier | Hypothesis under test | Verdict |
|---|---|---|
| **0** | Does M3 beat base on independent novel truth? | **Yes** — D1 P@5 0.335 vs 0.277; D2 R@20 0.791 vs 0.506. **M3-v1 is the deliverable.** |
| **1** | Is *label noise* the ceiling? (§3) | **No** — confirmed-only retrain is marginally worse everywhere. |
| **2** | Is the *task framing* the ceiling? | **No** — M3-R trains to AUC 0.90 but **ties base** on the anti-circular test. |

### Tier 2 — M3-R, and the finding that unifies the M3 line

**M3-R** reframes discovery as *candidate refinement*: the base model proposes
candidates, an XGBoost classifier reranks each as real cryptic site vs artifact
from the multimodal evidence. Its load-bearing design choice is
**base-score-matched hard negatives** — negatives are stratified to the positives'
base-score histogram, so the base score *cannot* be the discriminator and the
multimodal channels must earn their keep. By every training signal it worked:
held-out **AUC 0.903**, PR-AUC 0.73, and **56% of SHAP importance non-base**.

Yet on the same anti-circular D1/D2 test it **matches base to within ±0.002
everywhere**. Decomposing the AUC resolves the contradiction:

| AUC | Base | M3-R |
|---|---|---|
| Global (pooled across all candidates) | 0.585 | **0.675** (+0.09) |
| Within-gene (mean of per-gene AUCs) | 0.602 | 0.592 (−0.010) |

**M3-R's entire advantage is between-gene.** It learned *"this is the kind of
locus where cryptic sites occur"* — not *"this is the base within the locus that
is the real site."* Per-gene precision@k rewards only the latter.

**Root cause, and the reusable lesson:** the external multimodal tracks
(conservation, epigenetic, chromatin, RBP) are **gene/locus-level** — approximately
constant across the span of a gene's candidate positions. They carry *locus-level,
not position-level*, information.

This single finding **unifies every earlier "multimodal barely helps M3"
observation**: M3-v1's +0.016 genome-wide gain over its multimodal-zeroed variant,
the fact that zeroing multimodal is *slightly better* on D2 (R@20 0.844), and
Tier 2's train-0.90/eval-tie. All the same phenomenon. It also explains why the
same tracks *do* lift M1/M2 — canonical and alternative recognition is partly a
"what kind of locus" question; novel-site discovery is not.

### What this means for the design

The original §4 conclusion said the lever was "data and labels, not architecture."
The architecture half was right. The labels half was wrong. Revised:

> **Neither architecture nor label quality nor task framing is the lever. The
> lever is position-level features** — evidence that varies base-to-base
> (foundation-model scalars, local junction reads) rather than genome-averaged
> tracks — and **perturbation conditioning** ([M4](../M4/)), where the input
> genuinely differs per condition.

M3-R is not promoted, but it is not wasted: its global AUC 0.675 is repurposable
as **gene/locus-level triage** — ranking *which genes* to search, which is exactly
what it is good at. **M3-v1 remains the within-gene deliverable.**

### Trained artifacts

| Directory | What | Role |
|---|---|---|
| `output/meta_layer/m3_v1` | all 154,113 positives, confirmed 2× | **the deliverable** (registry `m3_v1`, `status: research`) |
| `output/meta_layer/m3s_v1_1_confirmed` | `--confirmed-only`, same architecture | Tier 1 — marginally worse |
| `output/meta_layer/m3r_candidate_refiner` | XGBoost refiner | Tier 2 — honest negative |

M3-v1 is inspectable per gene in the Bio Lab UI's **Novel Site Explorer**
(`/novel/{gene}`), restricted to the same held-out chromosomes evaluated here.

## Related
- [`m3_prerequisites.md`](m3_prerequisites.md) — the go/no-go decision and data-source survey.
- [`label_audit_A1.md`](label_audit_A1.md) — the cross-annotation experimental results.
- [`m3_training_data.md`](m3_training_data.md) — what a training example is, label pools, validation status, and the M3-R data.
- [`m3_eval_D_results.md`](m3_eval_D_results.md) · [`m3r_candidate_refiner_results.md`](m3r_candidate_refiner_results.md) — the Tier 0 / Tier 2 result tables.
- [`../meta_model_variants_m1_m4.md`](../meta_model_variants_m1_m4.md) — M1–M4 overview.
- [`../M4/`](../M4/) — the disease-conditional sibling.
- Published methods + results: [`docs/meta_layer/methods/06_m3_novel_site_formulation.md`](../../../../docs/meta_layer/methods/06_m3_novel_site_formulation.md) · [`docs/meta_layer/results/m3_novel.md`](../../../../docs/meta_layer/results/m3_novel.md)

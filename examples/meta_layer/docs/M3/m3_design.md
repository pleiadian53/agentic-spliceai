# M3 v1 — Design Synthesis

The design for **M3 v1**, the unconditional novel splice site discoverer.
This consolidates the scope decision, the label strategy, the architecture
position, and how the disease-context idea is incorporated. It is the public
companion to the prerequisites ([`m3_prerequisites.md`](m3_prerequisites.md))
and the experimental audit ([`label_audit_A1.md`](label_audit_A1.md)).

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

---

## 4. Architecture — the lever is data/labels, not architecture

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

- **v1 keeps the M2-S v3 backbone essentially as-is** (3-stream dilated CNN,
  cat-fusion, RF ≈ 400 bp, single 3-class head). Junction modality dropped
  from inputs (§2).
- The earlier "widen dilations to RF ≈ 2.6 kb" change is **demoted to an
  optional v1.1 experiment**, run *only if* a label-side baseline plateaus and
  diagnostics implicate context length — not as a default.
- Engineering effort goes to **label assembly and negative sampling**
  (Phase B), where the M2-S work says the gains actually live.

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

## 8. Status and next step

A1 (label audit) is **done** — see [`label_audit_A1.md`](label_audit_A1.md).
The next concrete step is **Phase B1**: apply the depth/breadth filter to the
65,163 survivors, fold in the Tier-1 catalogs with provenance tags, and emit
the positive pool. Then B2 (negatives) and B3 (long-read held-out).

## Related
- [`m3_prerequisites.md`](m3_prerequisites.md) — the go/no-go decision and data-source survey.
- [`label_audit_A1.md`](label_audit_A1.md) — the cross-annotation experimental results.
- [`../meta_model_variants_m1_m4.md`](../meta_model_variants_m1_m4.md) — M1–M4 overview.
- [`../M4/`](../M4/) — the disease-conditional sibling.

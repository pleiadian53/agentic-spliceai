# M4 — Perturbation-Induced Splicing: one conditional model, not many variants

**Status:** design note + first-instance validation (TDP-43/ALS). 2026-05-23.

M4 is the **conditional / counterfactual** arm of the meta layer: model a disease
as a *regulator perturbation* and predict the resulting **splicing change**.
It is distinct from its siblings, which are all *unconditional*:

| Variant | Question | Conditioning |
|---|---|---|
| M1-S | Where are the canonical splice sites? | none |
| M2-S | …and the alternative ones (beyond MANE)? | none |
| M3-S | …and *novel* ones (annotation-independent)? | none |
| **M4** | **What does perturbing regulator R do to splicing?** | **the perturbation R** |

This note records the design principle that keeps M4 general, why the naive
"a model per disease gene" approach is the wrong axis, what actually makes one
M4 work (data, not architecture), and what the TDP-43/ALS demo taught us.

---

## 1. The core principle: separate the *regulator* from the *target*

A perturbation that changes splicing has two sides:

- **Target** — the cryptic / aberrant exon that appears (UNC13A poison exon,
  STMN2 cryptic exon, an SF3B1 neo-3′SS, …). Targets are **open-ended**: there
  are thousands, gene by gene. They are the model's **output**.
- **Regulator** — the *trans*-acting cause that was perturbed (TDP-43, SF3B1,
  U2AF1, RBFOX, MBNL, hnRNPs, core spliceosome). Regulators are a **finite set**
  (~1,500 RBPs; far fewer dominant splicing regulators). They are the
  perturbation **input**.

> **Design rule:** model the **regulator → splicing map once**, and take *which
> regulator is perturbed* (and how much) as a **conditioning input** — never as
> architecture. One conditional M4; the disease is a different input, not a
> different model. The gene whose cryptic exon lights up is a *prediction*,
> never a parameter.

This is the same instinct that scoped M3 as unconditional ("don't let ~150
TDP-43 cryptic exons drive the architecture for the whole problem"). A bespoke
`tdp43` channel, or an `M4-UNC13A` variant, is the use-case-dependence trap:
it detects one story and generalizes to nothing.

```
   WRONG axis (combinatorial)              RIGHT axis (conditional)
   ──────────────────────────             ─────────────────────────
   M4_TDP43  → UNC13A, STMN2              ┌─────────────────────────┐
   M4_SF3B1  → ...                        │  one M4(features,        │
   M4_MBNL   → ...                        │        perturbation=R)   │
   M4_<gene> → ...   (∞ variants)         │  trained on many R's     │
                                          └─────────────────────────┘
                                            R = TDP-43 | SF3B1 | ...
```

---

## 2. Why one model generalizes (no combinatorial explosion)

The regulatory **grammar is shared across regulators**: an RBP acts through its
*sequence motif* + *position relative to the splice site* + *binding* +
*local context*. A model that learns this grammar from many regulators predicts
the effect of any regulator it has seen acting — TDP-43 (UG-rich intronic
binding ↦ repression) is one instance of the same code.

So **RBP-specificity should emerge from the motif/context, not from naming the
protein.** This matches the project's standing rule: *the same features serve
every meta-model; only the learned weights differ.* Represent the regulator
**generically**:

- today: the single `rbp_n_bound` count channel (all evidence, identity-blind —
  see §4) + the DNA sequence (which carries the motif);
- if needed: a **shared RBP embedding** or per-RBP channels (all RBPs as input
  dimensions, TARDBP just one of ~168) — **not** a hand-built `tdp43` channel.

An embedding additionally buys generalization to *held-out* regulators with
similar binding preferences — something per-protein channels cannot.

---

## 3. What actually makes one M4 work: perturbation-paired labels

The missing ingredient is **not architecture — it is paired training labels.**
To learn the *direction* of a perturbation, the model must see **both states**:

```
   regulator PRESENT  (WT)         regulator DEPLETED (KD/KO)
   target exon OFF         ──────►  target exon ON          = ΔPSI
   (the only state we                (the de-repressed state the
    have in normal-tissue              model must learn from)
    annotations today)
```

The data substrate that supplies this, across many regulators:

- **ENCODE shRNA-knockdown RNA-seq + ΔPSI** for ~200+ RBPs (K562/HepG2): the
  causal chain *RBP → aberrant binding → neo-splice site*, measured.
- disease-/tissue-specific KD where the generic panel doesn't reach (e.g.
  TDP-43 knockdown in a neuronal line for the ALS arc).

Train one conditional M4 on this multi-RBP KD→ΔPSI corpus; condition at
inference by **ablating the perturbed regulator's contribution** (the data-level
counterfactual, §4). Because training spans many regulators, the model learns
the general map and applies to any perturbation by changing the input — ALS
(TDP-43 loss), MDS (SF3B1), DM1 (MBNL sequestration), etc.

> **Cancer-induced framing (companion insight):** cancer-dysregulated RBP
> binding from K562/HepG2 eCLIP is a *liability* for M1/M2 canonical prediction
> but the **primary signal** for M4 — the signature {high RBP binding, low
> conservation, low junction breadth, dysregulated RBP} *defines* an induced
> neo-splice site. Do not filter "cancer artifacts" from M4 training; they are
> the positive examples. A generic `rbp_n_cancer_dysregulated` flag (planned)
> makes the signal explicit without naming a protein.

---

## 4. The current feature substrate and its limits

The RBP modality is **one channel, `rbp_n_bound`**, resolved internally like the
epigenetic/conservation channels (no per-script flags; harness all evidence by
default). It is a **count of overlapping eCLIP peaks** at each position
(`DenseFeatureExtractor._query_rbp`), drawn from the all-cell-line union
`data/mane/GRCh38/rbp_data/eclip_peaks_neuronal.parquet`
(ENCODE K562/HepG2 ∪ neuronal SH-SY5Y/H9 TARDBP).

Two consequences matter for M4:

1. **The channel is identity-blind.** It collapses *which* RBP and *which* cell
   line into a single scalar. The parquet keeps `rbp`/`cell_line`, but the model
   sees only "N peaks bound here," never "TDP-43 bound here." Overlapping peaks
   *stack* (the same RBP across 4 cell lines adds +4); `signal_value`/p-value are
   ignored. So the model cannot condition on a specific regulator *from the
   channel alone* — regulator identity must come from the sequence motif (today)
   or a regulator-resolved feature (future, §2).
2. **The counterfactual is done at the data level.** "Knock out regulator R" =
   drop `rbp == R` rows from the parquet and recompute the channel. This works
   precisely because the parquet retains identity even though the channel does
   not — the manipulation lives in the data, not in something the model must
   name.

---

## 5. First-instance validation: TDP-43 / ALS

The TDP-43 cryptic-splicing story is M4's **first validation instance** —
deliberately *not* the thing the architecture is built around.

**Biology.** TDP-43 (TARDBP) binding *represses* two well-characterized cryptic
events; they appear when TDP-43 is **lost** (ALS/FTD):
- **STMN2** (chr8,+): cryptic exon in intron 1 with a premature polyA →
  truncated, non-functional STMN2.
- **UNC13A** (chr19,−): a 128-bp "poison" cassette exon → frameshift/PTC → NMD.

(Curated GRCh38 coordinates: `examples/UI_integration/als_cryptic_sites.py`.
Genome-wide neuronal TDP-43 binding fetched via
`examples/UI_integration/fetch_encori_tardbp.py`.)

**What the demo showed** (`03_cryptic_site_detection`, `04_tdp43_ablation_counterfactual`,
`05_explainability_integrated_gradients`, `06_demo_synthesis`):

| Observation | Result | Reading |
|---|---|---|
| Detection of the UNC13A cryptic donor | base P(splice)=**0.000** → **M2-S ≈0.5** (over threshold) | M2-S *detects* a cryptic site the base model misses entirely |
| STMN2 cryptic acceptor | M2-S P≈**0.99** | detected |
| TDP-43 ablation counterfactual (KO − WT) | Δ ≈ **−0.04** (donor) — **wrong sign** | de-repression **not** learned |
| Integrated Gradients at the donor | **57% sequence / 43% multimodal**; `rbp_n_bound` top multimodal channel | the detection is mostly intrinsic splice grammar; RBP a real but secondary, *non-directional* signal |

**Root cause of the negative (two layers, both data not architecture):**
1. **Identity-blind channel** (§4): the model never sees "TDP-43," only a generic
   count — so it cannot learn a TDP-43-specific rule.
2. **No perturbed state in training:** labels are normal-tissue only (TDP-43
   present, cryptic OFF). With no de-repressed examples, there is nothing to
   learn the *sign* from — exactly the gap §3 fills.

**Conclusion.** Detection works from sequence + binding; the *mechanism*
(repression → de-repression on knockout) requires perturbation-paired labels and
(optionally) regulator-resolved features. This parallels the receptive-field
finding elsewhere in the project: the lever is data/labels, not architecture.
The honest negative is itself the result — and the spec for the next step.

---

## 6. Roadmap

1. **Labels:** assemble multi-RBP knockdown → ΔPSI (ENCODE shRNA-KD, ~200 RBPs)
   + a neuronal TDP-43 KD set for the ALS arc.
2. **Train one conditional M4** on that corpus; perturbation via data-level
   ablation of the regulator. Generic features only.
3. **(If warranted) regulator-resolved features:** RBP embedding / per-RBP
   channels — added because the data shows it helps, not by speculation.
4. **Validate de-repression:** knockout should *raise* P(splice) at TDP-43
   cryptic targets (the sign §5 could not produce).
5. **Generalize:** the same model + a different perturbation input → SF3B1,
   U2AF1 (cancer), MBNL (DM1), etc. No new variants.

## 7. Honest limits

- Generalizes **within the regulators and cell-type contexts it trained on**.
  TDP-43's *neuronal* cryptic targets are out-of-distribution for ENCODE
  K562/HepG2 KD data → that arm needs neuronal data, not a new model.
- Truly novel regulators / unseen contexts: motif embeddings extend reach but
  remain research-open.
- Annotation-derived labels carry circularity risk; validate against a held-out
  long-read truth set (see the meta-layer design notes).

## References

- Scripts: `examples/UI_integration/{03_cryptic_site_detection,
  04_tdp43_ablation_counterfactual, 05_explainability_integrated_gradients,
  06_demo_synthesis}.py`; `als_cryptic_sites.py`; `fetch_encori_tardbp.py`.
- Feature substrate: `splice_engine/features/dense_feature_extractor.py`
  (`_query_rbp`, the `rbp_n_bound` channel); RBP union parquet under
  `data/mane/GRCh38/rbp_data/`.
- Related docs: `../meta_model_variants_m1_m4.md`, `../m3_prerequisites.md`,
  `../ood_generalization.md`.

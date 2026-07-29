# M4 — Mutation-Induced Arm: Status, Results, and the Label-Granularity Ceiling

**Date:** 2026-07-29 · **Scope:** the *mutation-induced* half of M4 — score a variant, predict the
splicing change it causes, and (aspirationally) whether that change is pathogenic.

> **Not to be confused with the other M4 arm.** M4 has two halves that share a name and little else:
>
> | Arm | Perturbation | Label source | Status |
> |-----|--------------|--------------|--------|
> | **Regulator / knockdown** | knock down RBP *R* | ENCODE shRNA-KD rMATS → `kd_dpsi.parquet` | **3.17M measured per-site signed ΔPSI rows** ([`../../data_preparation/m4/`](../../data_preparation/m4/)) |
> | **Mutation-induced** (this doc) | a single nucleotide change | ClinVar / MutSpliceDB / SpliceVarDB | **label-starved — see §4** |
>
> The contrast between these two rows is the main finding of this document.

---

## 1. TL;DR

- Variant Δ-scoring **works and is shipped** (Phase 1A/1B): ref-vs-alt scoring, SpliceAI-convention
  `DS_DG/DL/AG/AL`, and a rule-based splice-consequence classifier.
- On **pathogenicity ranking the meta layer does not beat the base model** — ClinVar ROC-AUC
  **0.754 (base)** vs **0.753 (M2-S)**. That is not a tuning failure; it is architectural (§3).
- Where the meta layer *does* win is **consequence concordance** — *what kind* of splicing change —
  M2-S **72.1%** vs M1-S **50.7%** on MutSpliceDB.
- The historic "delta as a feature in a pathogenicity classifier" idea is **designed but never run**
  (ROADMAP Phase 8.3). The archived 2025 attempts at a learned variant classifier topped out at
  **ROC-AUC 0.58–0.61**.
- **The binding constraint is label granularity, not model capacity** (§4): we have essentially **zero**
  variants annotated with *measured per-nucleotide* splicing change. That is precisely what a
  per-nucleotide Δ model needs as supervision.

---

## 2. What was built and measured

### 2.1 Archive line (Dec 2025, SpliceVarDB labels — "is this variant splice-altering?")

Four experiments under [`docs/meta_layer/meta-spliceai-archive/experiments/`](../../../docs/meta_layer/meta-spliceai-archive/experiments/).
None of these touched pathogenicity — the labels were mechanistic (SpliceVarDB splice-altering vs normal).

| Exp | Approach | Result | Why it stopped |
|-----|----------|--------|----------------|
| 001 | Train on canonical sites, hope it transfers to variants | acc 99.11%, PR-AUC 0.987 — but variant detection **67% → 17%** | *"High classification accuracy on canonical sites does NOT translate to variant effect detection."* |
| 002 | Siamese ref/alt → predict the **base model's** Δ | Pearson **r = 0.38** | *"The target (base model delta) is the limiting factor"* — supervising on the base model's own errors |
| 003 | Binary "is it splice-altering?" | ROC-AUC **0.61**, F1 **0.53** | Gate was F1 > 0.7; never passed → the planned Steps 2–4 (type / where / how strong) were **never implemented** |
| 004 | "Validated delta" — force target to 0 when SpliceVarDB says *normal*, skip ambiguous | **r = 0.41** (best), ROC-AUC 0.58 | Best of the line, still unusable. Trained on **2,000** of ~50K available variants (laptop-only compute at the time) |

**The structural flaw in 004, stated plainly:** "validation" only corrects the *negatives* (forces them
to zero) and drops the ambiguous cases. For positives it still uses the **base model's** Δ magnitude.
So there was never an independent ground-truth Δ anywhere in the training signal.

### 2.2 Current line (2026, ClinVar + MutSpliceDB — Δ as a ranking score)

Pipeline: [`01_single_variant_delta.py`](../01_single_variant_delta.py) →
[`01b_splice_consequences.py`](../01b_splice_consequences.py) →
[`03_clinvar_benchmark.py`](../03_clinvar_benchmark.py) /
[`04_mutsplicedb_benchmark.py`](../04_mutsplicedb_benchmark.py).
Ranking score = `max_delta_within_radius(50)` (matches OpenSpliceAI's `dist_var=50`).

**ClinVar — pathogenic vs benign** (splice-filtered, N=2,059; 1,581 path / 478 benign; prevalence 76.8%):

| Model | ROC-AUC | PR-AUC | Sens@0.5 |
|-------|--------:|-------:|---------:|
| **Base (OpenSpliceAI)** | **0.754** | 0.924 | 0.476 |
| M1-S v2 | 0.734 | 0.916 | 0.418 |
| M2-S v2 | 0.753 | 0.923 | 0.386 |

Unfiltered ClinVar (N=11,310): base **0.666**, M1-S 0.662, M2-S 0.632.

!!! warning "Quote lift, not raw PR-AUC"
    PR-AUC's baseline **is the prevalence** (0.768 here), so 0.92 looks impressive and mostly is not.
    Lift over baseline: base **+0.254 ROC / +0.156 PR**. Raw PR-AUC is not comparable across benchmarks
    with different prevalences.

**MutSpliceDB — consequence concordance** (N=434 scored; RNA-seq-validated):

| Era | Checkpoints | M1-S | M2-S |
|-----|-------------|-----:|-----:|
| v2 sweep (2026-04-15) | `m1s_v2_logit_blend` / `m2s_v2` | 0.447 | 0.680 |
| **v4** | `m1s_v4_cleanannot` / `m2s_v4_cleanannot` | **0.5069** | **0.7212** |
| v4 + HGVS resolver | same | 0.5069 | 0.7212 (unchanged) |

The **+21 pt M2-vs-M1 gap** is the real signal here: the alternative-site model classifies *what kind*
of splicing change far better. Note the metric is lenient by construction — `intron_retention` matches
any of five predicted labels, while `exon_skipping` matches only itself.

---

## 3. Why the meta layer can't help Δ-ranking — the locus-cancellation result

For an SNV, the multimodal features are **identical between ref and alt**:

```python
# variant_runner.py
mm_alt = mm_ref.copy()   # identical for SNVs
```

Conservation, junction support, H3K36me3, ATAC, eCLIP peaks are properties of the **genomic locus**,
not of the variant. Swap one base and the ENCODE tracks do not move — so in the `alt − ref`
subtraction they **cancel to zero**. Multimodal evidence can shape *where* the max-|Δ| sits inside the
window, but carries **no variant-specific information**. For pathogenicity ranking via `max|Δ|`, the
multimodal stack is architecturally dead weight.

!!! abstract "The same finding as the M3-R negative"
    This is the variant-shaped face of the [M3-R result](../../../docs/meta_layer/results/m3_novel.md#m3-r-the-candidate-refiner-milestone).
    Locus-level features can say *what kind of locus* this is, never *which base* matters:

    | | Signal they carry | Signal the task needs | Outcome |
    |-|-------------------|----------------------|---------|
    | **M3-R** (novel sites) | between-gene (AUC 0.675) | within-gene ranking | ties base |
    | **Variants** (this arm) | locus-constant | ref-vs-alt difference | cancels to ~0 |

    One root cause, two arms. It also predicts where multimodal *does* pay: tasks scored **at a locus**
    rather than *within* or *across* one — which is exactly M1/M2 (canonical + alternative-site
    recognition), where the gains are large and real.

---

## 4. The real ceiling: label granularity

The models are asked to predict a **per-nucleotide** Δ. Ask what supervision exists at that grain:

| Corpus | Size | What the label actually is | Per-nucleotide measured Δ? |
|--------|-----:|----------------------------|:--:|
| **ClinVar** (splice-filtered) | 2,059 | *clinical* pathogenic / benign | ❌ no splicing info at all |
| **SpliceVarDB** | ~50K | variant-level "splice-altering / normal" | ❌ binary, no coordinates |
| **SpliceVault** | 144,094,769 rows | *if* this site is disrupted, the top-4 mis-splicing events observed across RNA-seq | ❌ a property of the **site**, not of the variant; no measured per-variant effect size |
| **MutSpliceDB** | **446 rows / 441 variants** | induced-site `chrom, position, strand, site_type` + `effect_type` ✅ *right grain* | ❌ categorical outcome, no ΔPSI magnitude |

**So the count of variants with a measured, coordinate-resolved, quantitative splicing outcome is
effectively zero.** MutSpliceDB is the only corpus with the right *shape* — it names the induced site
and its coordinate — and it holds **441 distinct variants**, of which **433 are intron retention** and
13 are `unknown`. There is no usable second class, let alone a magnitude to regress against.

That is the honest reason the archive line plateaued at r ≈ 0.38–0.41 and AUC ≈ 0.58–0.61. Experiment
002's own conclusion ("the target is the limiting factor") was the right diagnosis one level down: the
target was borrowed from the base model *because no measured target existed*.

### The contrast that makes the point

The regulator arm of M4 solved exactly this problem, in its own domain, this month:

| | Mutation-induced arm | Regulator/KD arm |
|-|----------------------|------------------|
| Perturbation | one nucleotide | knock down RBP *R* |
| Measured per-site signed ΔPSI | **~0** | **3,170,934 rows** |
| Regulators / conditions | — | 185 RBPs |
| Result | Δ-ranking ties base | corpus built; training formulation next |

Same conditional question — *"given this perturbation, what happens at this site?"* — but only one arm
has supervision at the output grain. **The blocker is data, not architecture.**

### What would actually move it

Ranked by how directly each supplies (variant → per-site quantitative splicing change):

1. **sQTL catalogs (GTEx / recount3).** Variant-linked splicing changes with coordinates *and* effect
   sizes, at scale. The closest available analogue to what the KD corpus did for regulators. Already
   named as a medium-term item in the sweep doc; **not yet acquired** (no sQTL data on disk).
2. **Saturation-mutagenesis / MPRA splicing assays** (Vex-seq, MFASS, saturation genome editing).
   Dense variant → ΔPSI at a small number of loci — ideal for *calibrating* magnitude, narrow in gene
   coverage. Also the natural fit for the planned saturation-mutagenesis application.
3. **Full SpliceVarDB (~50K)** instead of the 2,000-sample slice. Still binary, so it raises the
   ceiling on the *classification* framing only — worth it as a cheap upgrade, not a fix.
4. **Phase 8.3 clinical head** (stack `log(gnomAD AF)`, LOEUF/pLI, AlphaMissense, splice-Δ,
   consequence type). This is the *pathogenicity* question, and it is largely orthogonal to splicing —
   ~89% of ClinVar pathogenic variants act by non-splicing mechanisms, which is the PR-AUC floor no
   splice model can cross. Worth doing, but it improves the *clinical* answer, not the splicing one.

---

## 5. The HGVS / strand saga (resolved, with one caveat)

The original 2026-05-30 hypothesis — "MutSpliceDB positions are off by 1–13 bp for ~53% of rows" — was
**wrong**. The real defect was the **`strand` column**: the upstream MutSpliceDB CSV has **no strand
column at all**, and the parser fabricated one via a gene-name GTF lookup that missed ~53% of rows,
silently defaulting every miss to `+`.

Fixed by `HgvsResolver` (gffutils-backed, MANE): of 438 audited rows, **95.4% resolved**, **100%
FASTA-ref match**, **99.3% position-exact** (confirming positions were right all along), and strand
agreement went 46.7% → **100%**. Ref-mismatch warnings dropped from hundreds to **0**.

!!! warning "One claim in the repo is under-supported — do not repeat it as stated"
    The narrative says the identical before/after numbers *prove* "the consequence-concordance metric
    is robust to allele orientation." But
    [`04_mutsplicedb_benchmark.py`](../04_mutsplicedb_benchmark.py) resolves strand as
    `strand = gene_strands.get(gene, v.strand)` — **a GTF lookup that overrides the record's strand**.
    Spot-checking the four known-flipped genes (MTHFR, POMGNT1, FUBP1, GLUL), the *pre*-resolver run
    already used `−`. The benchmark was therefore **already immune** to the TSV bug, so the alleles
    likely never differed between the runs. The numbers are fine; the *explanation* is not established.
    Demonstrating robustness would require forcing the raw TSV strand through the scorer.

**Known residue:** 3 off-position rows (HLA-C −50 bp, suspected MANE-vs-source convention divergence in
the polymorphic HLA region; MET ×2 at +1 bp); 16 alt-isoform transcripts absent from MANE; 9 hg19 rows
still needing liftover.

---

## 6. Other documented limitations

- **Benchmark composition.** ~89% of ClinVar pathogenic variants are non-splicing mechanisms → a hard
  PR-AUC floor (~0.72 unfiltered) that no splice model can beat. MutSpliceDB is ~all intron retention,
  so the exon-skipping arm (M1-S 2/7, M2-S 1/7 in the v4 run) is statistically meaningless.
- **OOD genes.** Variants in genes thinly represented in MANE produce little signal — PYGB (paper
  Δ=0.77, ours 0.096), CDC25B (paper Δ=0.82, ours **0.005**).
- **No isoform fraction.** Δ says *whether* splicing changes, never *how much* — a Δ of 0.9 does not
  distinguish 10% from 90% of transcripts.
- **Reading-frame analysis is stubbed** — CDS extraction works, frame-preservation logic does not.
- **Calibration trade-off.** M2-S's softer temperature costs it at strict thresholds (detection @
  |Δ|≥0.8: M2-S 0.48 vs M1-S 0.75 vs base 0.76) → for strict-threshold confirmation, prefer the base model.
- **The meta layer cannot override a confident base "no splice" call**, which caps the OOD cases above.

---

## 7. Status

| Phase | Item | Status |
|-------|------|--------|
| 8.1A | Variant Δ scoring | ✅ Done |
| 8.1B | Splice-consequence classification | ✅ Done (reading-frame stubbed) |
| 8.2 | ClinVar + MutSpliceDB benchmarks + radius sweep | ✅ Done — **but ClinVar was never re-run on v4** (all ClinVar numbers are v2-era) |
| 8.3 | Clinical pathogenicity head | ⬜ Design-only, never implemented |
| 8.4 | Saturation mutagenesis / SpliceVarDB | ⬜ Planned |
| 8.5 | Agentic clinical interpretation | ⬜ Planned |

**Doc drift to be aware of:** [`../README.md`](../README.md) and
[`../docs/development_roadmap.md`](../docs/development_roadmap.md) still describe the April-2026 state
("M1-S v2", "Phase 2–3 planned") and predate the v4 + resolver eras entirely.

---

## Related

- [M3 novel-site results](../../../docs/meta_layer/results/m3_novel.md) — the between-gene/within-gene
  version of the same locus-level finding.
- [`m4_benchmark_sweep.md`](m4_benchmark_sweep.md) — the 2026-04-15 v2 sweep (source of the ClinVar numbers).
- [`variant_effect_validation.md`](variant_effect_validation.md) — 13 disease-gene variants + RNA-seq validation.
- [`../../meta_layer/docs/M4/m4_conditional_design.md`](../../meta_layer/docs/M4/m4_conditional_design.md) — the conditional-M4 design note.
- [`../../data_preparation/m4/`](../../data_preparation/m4/) — the regulator-arm ΔPSI corpus (the contrast in §4).
- [`negative_strand_and_variant_effects.md`](../../../docs/variant_analysis/negative_strand_and_variant_effects.md) — strand/coordinate traps.

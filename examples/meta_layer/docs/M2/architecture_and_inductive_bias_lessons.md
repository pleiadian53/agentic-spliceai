# M2-S: Architecture, Inductive Bias, and the Feature-Redundancy Finding

This note records what a sustained architecture-search effort on the M2-S
(alternative splice site) meta-layer taught us about **when fusion
architecture matters and when it doesn't**, and the feature-redundancy
analysis that explains why.

**One-line takeaway:** for M2-S, fusion architecture is not the lever.
Every mixing mechanism converges to the same ceiling because the
auxiliary modalities carry little discrimination orthogonal to the base
SpliceAI scores — and the small complementary signal that does exist is
linear and already captured by simple concatenation. The leverage is
upstream (features and labels), not in the fusion mechanism.

---

## 1. The architecture arc

M2-S underperforms M1-S (canonical) by a wide margin. The natural first
instinct was that a richer fusion architecture would close the gap. We
tested four distinct inductive biases, all at matched capacity (H=32):

| Fusion / spatial bias | Mechanism | Val PR-AUC |
|---|---|---|
| **cat-fusion (v3)** | concatenate streams → 1×1 conv → CNN; immediate full mixing | **0.807** |
| global cross-attention | per-position selective mixing, full sequence | 0.637 |
| local windowed cross-attention (W=256) | per-position selective mixing, ±128 bp | 0.802 |
| wider dilations (RF ~2.6 kb) | same fusion, much longer spatial context | ~0.80 (no gain) |

Three of four tie at ~0.80; global attention is *worse* (its symmetric
global access dilutes the rare positive class — see the attention
post-mortem). **The model is essentially invariant to how, and over what
range, the modalities are mixed.**

A fifth idea — **grouped Conv1d** (depthwise/grouped mixers: learn
within-modality spatial patterns before allowing cross-modality
interaction) — prompted the question: *would a staged-mixing inductive
bias help where immediate and selective mixing did not?* Rather than
spend another training run, we tested the **precondition** for grouped
conv to help, which is measurable without training.

---

## 2. The precondition: are the modalities complementary or redundant?

Grouped/staged mixing helps when modalities carry **orthogonal,
non-linear** signal that early cross-modality mixing would entangle and
lose. If the auxiliary modalities are either (a) redundant with the base
scores, or (b) complementary but linearly accessible, then staged mixing
buys nothing over concatenation.

We measured this directly on the full-genome feature set
(`12_feature_redundancy_analysis.py`), three tests:

### Test 1 — Cross-modality correlation (Spearman)

Mean |ρ| between base scores and each auxiliary modality:

| Modality | mean \|ρ\| with base | Reading |
|---|---|---|
| conservation | 0.20 | linearly independent |
| epigenetic | 0.16 | linearly independent |
| RBP | 0.13 | linearly independent |
| junction | 0.58 | correlated (it *is* RNA-seq evidence of splicing) |

Conservation, epigenetic, and RBP are linearly independent of the base
scores — so *in principle* there is room for complementary features.

### Test 2 — Where the base model is uncertain, do modalities rescue?

The crux. At TRUE splice sites where the base model assigns the correct
class < 0.5 (the sites M2 must rescue), can each modality discriminate
them from negatives? AUC by channel:

| Channel | AUC at base-uncertain sites | AUC at all true sites |
|---|---|---|
| phastcons (conservation) | **0.78** | 0.91 |
| phylop (conservation) | **0.75** | 0.87 |
| junction_log1p | **0.97** | 0.99 |
| junction_has_support | **0.97** | 0.99 |
| dnase (epigenetic) | 0.65 | 0.69 |
| atac (epigenetic) | 0.61 | 0.60 |
| h3k4me3 (epigenetic) | 0.60 | 0.54 |
| rbp_n_bound | 0.56 | 0.60 |
| rbp_n_sr_proteins | 0.50 | 0.51 |

Conservation and junction carry genuine rescue signal where the base
model fails. Epigenetic is weak-moderate. **RBP is weak (≈0.56)** — on
this set it does *not* light up where base scores are uncertain, contrary
to the motivating intuition (caveat in §4).

### Test 3 — Incremental discrimination beyond base scores

Logistic AUC, base scores alone vs. base + each modality group:

```
AUC(base only):            0.9969
base + conservation:  lift -0.0006   (redundant)
base + epigenetic:    lift -0.0002   (redundant)
base + junction:      lift +0.0003   (redundant)
base + rbp:           lift -0.0003   (redundant)
base + ALL modalities: lift +0.0007  (negligible)
```

Once you have the base scores, **no modality adds meaningful aggregate
discrimination.**

### Reconciling Test 2 and Test 3

These look contradictory (conservation/junction rescue uncertain sites,
yet add ~0 aggregate lift) but aren't: the base model is so good on this
set that only ~3.6% of true sites are base-uncertain. The rescue signal
is real but applies to a small slice, so it barely moves the aggregate
metric.

There's a second reason the lift is ~0, and it's the one that bears on
grouped conv. **Test 3 is a *linear* probe (logistic regression) — the
weakest model you can put on the features, and therefore a lower bound on
what *any* model can extract.** It finds that the modalities add no
incremental discrimination beyond the base scores. The rescue signal that
does exist (Test 2) shows up at the **single-channel** level — one
channel, one threshold reaches AUC 0.75–0.97 — i.e. it is low-dimensional
and readily accessible to even a simple model, not hidden in channel
*combinations* that only become visible after dedicated within-modality
processing. So a linear lower bound already finds the ceiling.

---

## 3. Verdict on grouped Conv1d

**Unlikely to help — and now we know why, without spending a training run.**

First, a precision point on what grouped conv is. It does **not** add
representational power: a standard conv (`groups=1`) can represent
everything a grouped conv can, and more — grouped conv is a strict
*subset*. Likewise, cat-fusion (`concat → 1×1 conv → nonlinear CNN`) is a
universal approximator and can already represent non-linear cross-modality
interactions. So the question is never "which architecture *can* capture
the relationship" — both can. Grouped conv is a **regularizing inductive
bias**: it forces within-modality feature learning before cross-modality
mixing, which improves *generalization* only when there is complementary
signal that naive early mixing would entangle, suppress, or overfit.

- The modalities are *not* redundant in the correlation sense (Test 1),
  and a couple carry single-channel rescue signal where the base model
  fails (Test 2). But **Test 3's linear probe — a lower bound — already
  finds zero incremental discrimination beyond the base scores.** Grouped
  conv's job is to *protect* hard-to-learn complementary signal; with no
  incremental signal to protect in this regime, the prior has nothing to
  exploit. You cannot regularize your way to signal the features don't
  contain.
- The single modality with strong rescue signal (junction) is
  near-circular — it is RNA-seq evidence of splicing, i.e. almost a
  label. (This is precisely why M3 uses junction support as a *label*,
  not an input.)
- RBP — the original motivation for staged mixing — is weak on this set.

This is consistent with the architecture arc: cat-fusion, attention, and
wider context all hit ~0.80 because they all answer *"how do I mix the
modalities?"* when the binding constraint is *"how much orthogonal signal
is there to mix?"* — and the answer is: not much, on this feature set, in
this regime.

---

## 4. Caveats (what this analysis does and doesn't settle)

1. **Regime: this ran on MANE canonical sites, not M2 alternative sites.**
   The base model (OpenSpliceAI, MANE-trained) is near-perfect on
   canonical sites (AUC 0.997; only ~3.6% base-uncertain). M2's actual
   task is Ensembl *alternative* sites, where the base model is genuinely
   uncertain over a much larger fraction — so the rescue headroom there is
   bigger. **The canonical-regime result is a lower bound on the M2
   opportunity.** The #1 follow-up is to rerun this analysis on the
   Ensembl alternative-site feature parquets (not available locally at the
   time of writing). The *directional* findings (conservation
   complementary; junction near-label; RBP weak) should transfer.
2. **RBP cell-line mismatch.** The eCLIP features are ENCODE K562/HepG2.
   Many alternative splicing events (e.g. neuronal TDP-43 cryptic exons)
   are driven by RBPs in cell contexts these don't cover. RBP being weak
   here may reflect the wrong cell lines, not that RBP is intrinsically
   uninformative. Enriching RBP with neuronal CLIP is a feature-side
   lever, not an architecture one.
3. **Linear probes.** Tests 2–3 use single-channel AUC and logistic
   regression — they detect linear/threshold signal. A deep model could
   exploit non-linear structure they miss. But the architecture arc
   already provides the empirical check: deep models with various fusion
   schedules did *not* beat cat-fusion, consistent with the linear-probe
   conclusion.

---

## 5. The broader lesson — when fusion architecture is (not) the lever

Fusion architecture matters when complementary, non-linearly-entangled
modalities need careful mixing. It does **not** matter when:

1. **One stream already dominates.** If the base signal alone nearly
   solves the task (canonical: AUC 0.997), there is no headroom for any
   fusion scheme to exploit.
2. **Complementary signal is low-dimensional and single-channel-accessible.**
   If the useful auxiliary signal is exposed by a few channels via simple
   thresholds (so a linear probe already extracts it), then it is not
   hidden in channel combinations that a staged-mixing prior would need to
   protect; any reasonable model captures it and fancier mixing is redundant.
3. **The orthogonal information simply isn't in the features.** No mixing
   mechanism creates signal that the feature set doesn't contain.

For M2-S, the leverage is **upstream of fusion**: richer features
(neuronal RBP CLIP rather than K562/HepG2), better labels (PSI-weighted
continuous junction support rather than binary), and tissue conditioning.
These change the *information content* the model has to work with — which
is the actual constraint — rather than the *mixing mechanism*, which we
have now shown four times over is not.

**Practical rule of thumb that came out of this:** before investing
compute in a new fusion architecture, measure the precondition — the
incremental discrimination each modality adds beyond the dominant stream,
especially in the regime where the dominant stream fails. It is cheap,
local, and decisive.

---

## Reproduce

```bash
python examples/meta_layer/12_feature_redundancy_analysis.py \
    --chromosomes chr22 chr21 chr19 --base-uncertain-threshold 0.5
```

Substrate: `data/mane/GRCh38/openspliceai_eval/analysis_sequences/analysis_sequences_chr*.parquet`
(116-column Phase-5A feature tables). The attention-architecture results
referenced in §1 are recorded in the M2 alternative-site predictor's
dev notes.

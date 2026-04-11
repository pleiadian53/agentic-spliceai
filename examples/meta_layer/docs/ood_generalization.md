# Out-of-Distribution Generalization in Splice Site Prediction

How a model trained on one set of splice sites generalizes (or fails to
generalize) to splice sites it has never seen, and what this tells us
about the architecture and training data.

---

## 1. The Setup

The M1-S meta-layer is trained on **MANE** splice sites — one canonical
transcript per gene, ~370K sites, representing the most well-supported
isoform per locus.  But alternative splicing means most genes use
additional splice sites not in MANE, present in richer annotations like
Ensembl (~10M sites) and GENCODE (~3.5M sites).

These alternative sites are **out-of-distribution (OOD)** for the
meta-layer:
- They were never positive labels during training
- They occur in sequence contexts the model may not have learned to
  recognize as splice-like
- They may have weaker base model scores (the base model was also
  trained primarily on canonical sites)

The M2 evaluation asks: when we evaluate M1-S at these unseen sites,
does the meta-layer help or hurt?

---

## 2. The OOD Failure Mode (v1)

The v1 M1-S meta-layer **hurt** performance on alternative sites:

```
Alternative sites (Ensembl \ MANE):
  Base model PR-AUC:  0.749
  Meta model PR-AUC:  0.704   ← worse than base alone
```

The meta-layer, which dramatically improves canonical site prediction
(PR-AUC 0.98 → 0.999), makes things worse when evaluated on sites
outside its training distribution.  This is a textbook OOD failure.

### Why did it fail?

The meta-layer learns a **decision function** from three input streams:
DNA sequence, base model scores, and multimodal features.  During
training, it sees:

- Positive examples: MANE canonical splice sites (strong GT/AG motifs,
  high base model scores, strong conservation, junction support)
- Negative examples: everything else (including alternative splice
  sites, which are labeled as "neither")

The model learns that a "real" splice site has:
- High base model score (>0.9)
- Strong conservation
- Junction support
- Clear sequence motif

Alternative splice sites often have:
- Moderate base model scores (0.3-0.7)
- Weaker conservation (less constrained across species)
- Variable junction support (tissue-specific)
- Non-canonical or degenerate motifs

The meta-layer, having learned that strong signals = splice site and
moderate signals = background, actively suppresses the moderate-signal
alternative sites.  It's doing exactly what it was trained to do — the
problem is that its training distribution doesn't represent the full
diversity of splice sites.

### The architecture amplified the problem

The v1 probability-space blend had a **double-softmax bug** that
flattened the base model's signal.  A base model score of 0.65 at an
alternative site was compressed to ~0.40 after double-softmax, then
blended 50/50 with the meta-CNN's output (which had learned to call
this a "neither" position).  The result: the base model's moderate but
informative signal was overridden by the meta-layer's confident but
wrong classification.

---

## 3. The Fix: Preserving Base Model Signal (v2)

The v2 logit-space blend **reversed** the OOD failure:

```
Alternative sites (Ensembl \ MANE):
  Base model PR-AUC:  0.749
  Meta model PR-AUC:  0.775   ← now better than base
```

The logit-space blend operates as:

```
blended = alpha * meta_logits + (1 - alpha) * log(base_probs)
```

This preserves the base model's signal at its original scale.  When the
meta-CNN has low confidence (it hasn't seen this type of site), its
logits are near zero and the base model's contribution dominates.  When
the meta-CNN has high confidence (canonical sites it trained on), it
adds signal on top of the base.

The product-of-experts property means:
- **In-distribution** (canonical sites): both sources agree → sharp,
  confident prediction → better than either alone
- **Out-of-distribution** (alternative sites): meta-CNN is uncertain →
  falls back to base model → at least as good as base alone, potentially
  better if multimodal features add context

This is a form of **graceful degradation** — the model doesn't hurt
when it doesn't know, and helps when it does.

---

## 4. General Principles of OOD Generalization

### 4.1. Training distribution defines the decision boundary

A model's decision boundary is shaped by its training data.  Points
inside the training distribution are classified well; points outside
may be classified arbitrarily.  For splice site prediction:

```
Training distribution:    canonical GT-AG sites with strong signals
                          ↓
Learned boundary:         "splice if score > 0.8 AND conservation > X"
                          ↓
OOD behavior:             alternative sites with score 0.4 → "not splice"
```

The model isn't wrong about what it learned — it's wrong about what
it was asked to evaluate on.

### 4.2. Overconfident models fail worse OOD

A well-calibrated model that assigns p=0.6 to an uncertain prediction
is informative — it tells you the model is unsure.  An overconfident
model that assigns p=0.95 or p=0.05 to the same input gives you no
signal that it's operating OOD.

Temperature scaling and calibration help here.  The v2 model's learned
per-class temperature produces more calibrated outputs, which means the
blend weight between meta-CNN and base model is more meaningful at OOD
points.

### 4.3. Ensemble and blend architectures are naturally robust to OOD

The residual blend `alpha * meta + (1-alpha) * base` is a form of
**model ensemble**.  Ensembles are well-known to be more robust to
distribution shift than single models (Lakshminarayanan et al., 2017).

The key property: when the specialized model (meta-CNN) is uncertain,
the generalist model (base) takes over.  This is the "default to the
prior" behavior that Bayesian approaches also exhibit.

The logit-space formulation is better than probability-space for this
because:
- In logit space, uncertainty manifests as logits near zero (low
  magnitude), which contribute little to the blend
- In probability space, uncertainty manifests as probabilities near
  1/C (uniform), which still pull the blend toward uniform via the
  softmax

### 4.4. Label coverage determines generalization ceiling

No amount of architectural cleverness can make a model predict patterns
it has never seen if those patterns are fundamentally different from
training.  The meta-layer's OOD performance is bounded by:

1. **Base model quality** at OOD sites (the fallback)
2. **Feature informativeness** at OOD sites (can multimodal features
   provide signal even without label supervision?)
3. **Similarity to training distribution** (alternative sites that
   resemble canonical sites are easier to generalize to)

This motivates M2-S training: if OOD performance matters, expand
the training distribution to include alternative sites.

---

## 5. Measuring OOD Performance

### The set-difference protocol

For splice site prediction, we have a natural hierarchy of annotations:

```
MANE ⊂ GENCODE ⊂ Ensembl
```

The OOD set is the **set difference**: sites in the richer annotation
that are absent from the training annotation.

```
OOD sites = Ensembl \ MANE   (Eval-Ensembl-Alt)
OOD sites = GENCODE \ MANE   (Eval-GENCODE-Alt)
```

This is cleaner than typical OOD benchmarks (CIFAR → ImageNet, etc.)
because:
- The OOD data comes from the same genome (same sequences, same features)
- The only difference is whether the site was labeled as positive during
  training
- We can stratify by annotation tier (well-supported vs computational
  predictions) to understand which types of OOD sites are hardest

### What to measure

| Metric | What it tells you |
|--------|------------------|
| PR-AUC (meta vs base) | Does the meta-layer help or hurt OOD? |
| FN count at OOD sites | How many alternative sites does the model miss? |
| Per-tier breakdown | Which confidence levels of alt sites are hardest? |
| Score distribution at OOD sites | Is the model uncertain (calibrated) or confidently wrong? |

### The "does it hurt?" test

The simplest OOD diagnostic: compare `meta_PR-AUC` vs `base_PR-AUC` at
OOD sites.  If meta < base, the additional model complexity is
counterproductive on unseen data — the model is **overfitting to the
training distribution's decision boundary**.

For M1-S: v1 failed this test (0.704 < 0.749), v2 passes (0.775 > 0.749).

---

## 6. Strategies for Improving OOD Generalization

| Strategy | Complexity | M2 variant | Effect |
|----------|-----------|------------|--------|
| Fix the architecture (logit blend) | Low | v2 | Preserve base signal at OOD points |
| Expand training labels | Medium | M2-S | Include alternative sites in training |
| Confidence-weighted labels | Medium | M2-S/M2d | Downweight noisy OOD labels |
| Junction-validated labels | Medium | M2d | Use empirical evidence as label quality |
| Tissue conditioning | High | M2e | Match features to expression context |
| Calibration | Low | Post-hoc T | Make uncertainty estimates meaningful |

The progression Eval-Ensembl-Alt → Eval-GENCODE-Alt → M2-S → M2d represents increasing
investment in OOD robustness, with each step informed by the previous
step's empirical results.

---

## 7. Lessons for Other Domains

The OOD failure pattern observed here — specialized model hurts on
unseen data by overriding a more general model's signal — is common in:

- **Clinical ML**: A model trained on one hospital's data fails at
  another hospital with different patient demographics
- **NLP**: A sentiment model trained on product reviews fails on
  social media text
- **Drug discovery**: A binding affinity model trained on known
  drug-target pairs fails on novel targets

The architectural lesson (logit-space blending with graceful degradation
to the generalist model) applies broadly.  When you combine a specialist
with a generalist, ensure the specialist can *abstain* rather than
override the generalist when it encounters unfamiliar inputs.

---

## References

- Lakshminarayanan et al. (2017). "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles." NeurIPS.
- Ovadia et al. (2019). "Can You Trust Your Model's Uncertainty?
  Evaluating Predictive Uncertainty Under Dataset Shift." NeurIPS.
- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
- Hendrycks & Gimpel (2017). "A Baseline for Detecting Misclassified
  and Out-of-Distribution Examples in Neural Networks." ICLR.

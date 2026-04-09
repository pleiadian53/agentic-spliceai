# Temperature Scaling for Splice Site Prediction

Temperature scaling adjusts the confidence of model predictions without
changing the ranking.  It comes in two forms: **post-hoc** (applied
after training) and **learned** (integrated into the model).

---

## 1. What Temperature Does

Given raw logits `z` from a model, softmax produces probabilities:

```
p_i = exp(z_i) / Σ exp(z_j)
```

Temperature scaling divides logits by T before softmax:

```
p_i = exp(z_i / T) / Σ exp(z_j / T)
```

- **T > 1**: softens probabilities (less confident, more uniform)
- **T < 1**: sharpens probabilities (more confident, more peaked)
- **T = 1**: no change (standard softmax)

Temperature preserves the **ranking** of predictions — if class A had the
highest probability before scaling, it still does after.  Only the
magnitude of the probabilities changes.

---

## 2. Why Models Need Calibration

Modern deep networks are systematically **overconfident** (Guo et al.,
2017).  A model predicting p=0.95 for "donor" is not correct 95% of
the time — it might be correct only 80% of the time.

For splice site prediction, overconfidence manifests as **excess false
positives**: background positions that cross the classification threshold
because their splice-site probability is too high.  In the M1-S model,
15,883 FPs were observed before calibration.

Temperature scaling addresses this by softening the probability
distribution, pulling overconfident predictions back toward the
decision boundary.

---

## 3. Post-Hoc Temperature Scaling (Standard)

The traditional approach, applied **after training is complete**:

### Protocol

```
Train set   →  train model weights (T=1, standard softmax)
Val set     →  fit T by minimizing NLL (seconds, single parameter)
Test set    →  evaluate with fixed T (model weights frozen)
```

### Scalar temperature

One T for all classes.  Simple but limited — it sharpens or softens
all classes equally, which may not be appropriate when classes have
different calibration needs.

For M1-S, scalar T=0.35 made FPs **worse** (17,649 vs 15,883) because
sharpening the "neither" class also sharpened splice-site predictions.

### Class-wise temperature (OpenSpliceAI approach)

A vector `T = [T_donor, T_acceptor, T_neither]`, one per class:

```
p_i = exp(z_i / T_i) / Σ exp(z_j / T_j)
```

This allows different calibration per class.  For M1-S post-hoc
calibration:

```
T = [0.81, 0.81, 1.17]
     ^^^^  ^^^^  ^^^^
     sharpen splice    soften background
```

Result: FPs reduced from 15,883 → 14,195 with minimal recall loss.
Splice classes get sharpened (more decisive), background gets softened
(less overconfident).

### Implementation

```python
from agentic_spliceai.splice_engine.eval.streaming_metrics import TemperatureScaler

scaler = TemperatureScaler()
for gene_data in val_genes:
    logits = infer_full_gene(model, gene_data, return_logits=True)
    scaler.collect(logits, gene_data["base_scores"], gene_data["labels"])

result = scaler.fit(blend_alpha=0.5, blend_mode="logit")
T = result["temperature"]  # np.ndarray [3]
```

---

## 4. Learned Temperature (Integrated into Training)

The M1-S v2 model (logit-space blend) takes a different approach: the
per-class temperature is a **learnable parameter** trained end-to-end
alongside the model weights.

### Architecture

The model's output layer applies:

```
output = softmax((α × meta_logits + (1-α) × base_logits) / T)
```

where `T = [T_donor, T_acceptor, T_neither]` is an `nn.Parameter`
initialized to `[1.0, 1.0, 1.0]` and clamped to `[0.05, 5.0]`.

During training, `T` receives gradients through the cross-entropy loss
and adapts alongside the model weights.  The model learns its own
calibration.

### Why this works

Post-hoc temperature fixes calibration **after** the model has already
learned miscalibrated internal representations.  Learned temperature
lets the model adjust its confidence **during** learning, which can lead
to better-calibrated internal features.

The cross-entropy loss directly incentivizes calibration: a model that
assigns p=0.95 to a class that's correct 95% of the time achieves
lower NLL than one that assigns p=0.99 (overconfident) or p=0.80
(underconfident).  Learned T gives the model an extra degree of freedom
to achieve this calibration.

### Early results (M1-S v2, epoch 2)

```
T = [1.48, 1.13, 1.12]    α = 0.505
PR-AUC = 0.9901 (matches v1 best at epoch 46)
```

All T values are > 1 (softening), which differs from the post-hoc
result (T_splice < 1, sharpening).  This is expected: during training,
softer probabilities produce better-conditioned gradients.  The model
may sharpen T later as it converges, or the learned T may settle at
different values than post-hoc T because the model weights co-adapt.

### When post-hoc calibration is still useful

Even with learned temperature, post-hoc calibration may add value:

- **Distribution shift**: if the test data distribution differs from
  training (e.g., evaluating on Ensembl genes when trained on MANE),
  the learned T may be miscalibrated for the new distribution.
- **Threshold selection**: for a specific clinical application (e.g.,
  "minimize FPs at >99% recall"), post-hoc T can be re-optimized for
  that specific operating point.
- **Comparison**: post-hoc T on the new model provides an apples-to-
  apples comparison with the v1 model's post-hoc results.

---

## 5. Temperature and Variant Analysis

Temperature has a direct impact on variant delta scores:

```
delta = softmax(alt_logits / T) - softmax(ref_logits / T)
```

- **T > 1** (softer): smaller absolute deltas, fewer detected events.
  Lower sensitivity but fewer false alarms.
- **T < 1** (sharper): larger absolute deltas, more detected events.
  Higher sensitivity but more noise.

The M1-S v1 model had dampened variant deltas (1.5-5x weaker than the
base model) partly because of the probability-space blend bug, but also
because the residual blend inherently smooths predictions.  The v2
logit-space blend with learned T should produce sharper deltas because
the blend happens before softmax, preserving the full dynamic range.

---

## 6. Summary

| Approach | When T is optimized | Degrees of freedom | Changes model weights? |
|----------|--------------------|--------------------|----------------------|
| **Scalar post-hoc** | After training | 1 | No |
| **Class-wise post-hoc** | After training | num_classes | No |
| **Learned (M1-S v2)** | During training | num_classes | Co-adapted |

For new models, prefer learned temperature — it's strictly more
expressive and costs nothing (num_classes extra parameters).  Post-hoc
calibration remains available as a safety net for distribution shift or
application-specific tuning.

---

## 7. References

- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
  Establishes that modern DNNs are miscalibrated and proposes temperature
  scaling.
- Chao et al. (2025). "OpenSpliceAI improves the prediction of variant
  effects on mRNA splicing." Uses class-wise temperature scaling
  (per-class T vector) with Adam optimizer.
- Platt (1999). "Probabilistic Outputs for Support Vector Machines."
  The predecessor to temperature scaling (Platt scaling for SVMs).

# M1 — Canonical Splice Sites

**Task.** Refine the base model's predictions at the **curated canonical splice sites** (MANE). This is
the "sharpen what's already mostly right" regime — the base model already scores canonical sites well,
so the bar is high and the headroom is small. Two models cover it: **M1-P** (position-level XGBoost, the
reference baseline) and **M1-S** (the promoted sequence CNN).

**Bottom line:** the canonical task is essentially solved. M1-S reaches a macro PR-AUC of **0.9998**,
and the interesting result is not the top-line number but *which modality* delivers the remaining error
reduction — the splice-junction evidence.

---

## M1-P — position-level baseline

A 3-class XGBoost classifier over the tabular per-position feature table — the position-level (`-P`)
reference the sequence model is compared against.

**Evaluation.** SpliceAI chromosome holdout (Jaganathan et al. 2019); test chromosomes 1, 3, 5, 7, 9 =
5,515 genes / 1,993,415 positions (6.27 M sampled positions total). Counts below are at the model's
argmax operating point.

| Feature set | Accuracy | PR-AUC donor | PR-AUC acceptor | FN | FP |
|-------------|----------|--------------|-----------------|----|----|
| Base scores only (50 feat) | 99.22% | 0.9924 | 0.9890 | 1,468 | 14,032 |
| **Full stack (103 feat)** | **99.74%** | **0.9986** | **0.9972** | **555** | **4,547** |

Adding the full multimodal stack on top of the base scores **reduces errors by 62% (FN) and 68% (FP)**
(total FN −913, FP −9,485; donor FP −74%, acceptor FP −63%).

**Where the reduction comes from — junction evidence.** Adding the splice-junction modality alone
takes FN 1,061 → 555 (−48%) and FP 8,960 → 4,547 (−49%). By feature importance, `junction_has_support`
is the #3 feature by gain (0.248), behind two base-score-derived features (`type_signal_difference`
0.402, `acceptor_prob` 0.298).

!!! note "Gain underestimates conservation"
    Measured by **SHAP**, conservation features are ~10× more important than their **gain** ranking
    suggests. Use SHAP (or XGBoost `pred_contribs`), not gain, when judging modality contribution here.

**Conclusion.** The canonical task is near-saturated, the strongest signal beyond the base scores is
junction support, and the meta layer removes errors without introducing net false positives.

---

## M1-S — sequence model (current)

The promoted canonical model: a ~367K-parameter three-stream dilated CNN over base scores + sequence +
the multimodal channels. Version `m1s_v4_cleanannot`. Trained per
[Workflow Stage 4](../../workflows/meta_layer/04_training_m1s.md), evaluated per
[Stage 6](../../workflows/meta_layer/06_evaluation.md).

**Headline (SpliceAI holdout, MANE):**

| | Base (OpenSpliceAI) | M1-S (v4) |
|-|---------------------|-----------|
| Macro PR-AUC | 0.9986 | **0.9998** |
| Donor / acceptor PR-AUC | — | 0.99969 / 0.99974 |
| At F1-optimal: recall / FP / FN | — | **0.997 / 346 / 279** |

Validation macro PR-AUC was 0.99835 at the best epoch (epoch 8).

!!! warning "Operating point matters"
    At the naïve argmax threshold the same run reports far more false positives (~25 K) than at the
    F1-optimal point (346). That is a threshold artifact under extreme class imbalance, not a ranking
    failure — which is exactly why the headline is PR-AUC and the counts are quoted at F1-optimal.

### Modality contribution (ablation)

A leave-one-out ablation of the M1-S channels (its own eval universe, base OpenSpliceAI PR-AUC 0.9839
in that study) isolates where the refinement comes from, ranked by false-negative reduction vs base:

| Configuration | PR-AUC | FN reduction vs base |
|---------------|--------|----------------------|
| Full model (9 channels) | 0.9993 | **+93.0%** |
| − multimodal (sequence + base only) | 0.9962 | +75.1% |
| − junction support | 0.9955 | +50.6% |

- The **sequence CNN foundation alone** accounts for +75.1% FN reduction.
- **Junction is the single largest modality**, worth **+42.4 percentage points** of FN reduction.
- Conservation adds +5.7 pp; **epigenetic is slightly negative** (−1.2 pp — removing it *improves* FN
  reduction); RBP and chromatin accessibility are < 0.5 pp ("noise at this scale").

This mirrors the M1-P finding: beyond the base scores and the sequence backbone, junction support is
the decisive evidence for canonical sites.

### Version history

Every entry below is the **same architecture** (`concat_fusion`). What changed was the blend
hyperparameter, then the training corpus — two different axes, which the old `vN` labels did not
distinguish. Axis names in parentheses.

| Step | Axis that moved | What changed | Result |
|------|-----------------|--------------|--------|
| prob blend | hyperparameter | probability-space blend of base + meta | val PR-AUC 0.9899, test 0.9994 |
| logit blend | hyperparameter | learned logit-space blend (α = 0.535, per-class T = [1.18, 0.94, 1.14]) | val PR-AUC 0.9954, test 0.9996; FN 643, FP 13,427; recovers 93–95% of base signal on MYBPC3 donor-loss (prob blend: 68–71%) |
| `encode_rbp` → `neuronal_rbp` | **corpus** | RBP channel widened to the neuronal union | superseded before promotion |
| **`neuronal_rbp` → `cleanannot`** | **corpus** | minus-strand-corrected annotation | **current promoted model** — macro PR-AUC 0.9986 → 0.9998 |

Blend mode is a `config.pt` field, not a version — every checkpoint on disk now uses `logit`.
The promoted model's canonical ID is **`m1s.concat_fusion.cleanannot`**; its directory keeps the
historical name `m1s_v4_cleanannot`, where the `v4` is the corpus generation and never referred to
the `xattn_fusion` architecture. See the
[naming convention](../methods/naming_convention.md) and the
[workflow overview](../../workflows/meta_layer/README.md#3-three-independent-config-levers).

---

## Takeaways

- The canonical task is effectively saturated (macro PR-AUC 0.9998); improvements are now measured in
  hundreds of avoided errors, not PR-AUC points.
- **Junction support is the load-bearing modality** for canonical refinement, in both the position-level
  and sequence models.
- Epigenetic / RBP / chromatin channels contribute little to *canonical* sites — a finding that turns
  out to matter a great deal for [M3](m3_novel.md), where the same tracks are asked to do more.

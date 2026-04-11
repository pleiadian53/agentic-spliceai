# M1-P Full-Genome Results: Position-Level Meta-Layer Baseline

**Date**: 2026-03-31
**Model variant**: M1-P (position-level XGBoost, 3-class: donor / acceptor / neither)
**Base model**: OpenSpliceAI (GRCh38 / MANE)
**Split**: SpliceAI chromosome holdout (Jaganathan et al., 2019)
**Data**: 6.27M sampled positions across 24 chromosomes, 19,174 genes

> **Note**: This is the position-level (M1-P) baseline. For the sequence-level
> model (M1-S), see `m1s_ablation_study.md` and the evaluation results in
> `output/meta_layer/m1s/`.

---

## Headline Results

The multimodal meta-layer reduces splice site prediction errors by **62% (FN)
and 68% (FP)** compared to base model scores alone, demonstrating that
conservation, epigenetic, junction, and other evidence modalities carry
substantial information beyond what the base model captures.

---

## Experimental Setup

### Chromosome Split

Following the SpliceAI convention (Jaganathan et al., 2019):

| Set | Chromosomes | Genes | Positions |
|-----|------------|-------|-----------|
| Train | chr2,4,6,8,10-22,X,Y | 12,294 | 3,851,063 |
| Val | 10% holdout from train genes | 1,365 | 420,736 |
| Test | chr1,3,5,7,9 | 5,515 | 1,993,415 |

Val set used for XGBoost early stopping only.  Test set never seen during
training.

### Feature Set

103 numeric features across 9 modalities:

| Modality | Columns | Description |
|----------|---------|-------------|
| base_scores | 43 | OpenSpliceAI scores + derived features |
| genomic | 4 | Positional features, GC content |
| conservation | 9 | PhyloP, PhastCons + context statistics |
| epigenetic | 12 | H3K36me3, H3K4me3 (ENCODE, 5 cell types) |
| junction | 12 | GTEx v8 RNA-seq (353K junctions, 54 tissues) |
| rbp_eclip | 8 | ENCODE eCLIP RBP binding (K562, HepG2) |
| chrom_access | 12 | ATAC-seq + DNase-seq accessibility |
| sequence | (excluded) | 501nt DNA window (used only by deep models) |
| fm_embeddings | (pending) | Foundation model PCA features (GPU extraction) |

### XGBoost Configuration

- `n_estimators=500`, `max_depth=6`, `learning_rate=0.1`
- Inverse-frequency class weighting
- `tree_method="hist"`, `early_stopping_rounds=20`
- Training time: ~7.5 min on M1 MacBook (16 GB)

---

## Ablation: How Much Does Each Modality Help?

### Summary Table

| Configuration | Features | Accuracy | PR-AUC (D) | PR-AUC (A) | FN | FP |
|--------------|----------|----------|-----------|-----------|------|-------|
| Base scores only | 50 | 99.22% | 0.9924 | 0.9890 | 1,468 | 14,032 |
| + conservation | 59 | 99.46% | 0.9965 | 0.9937 | 974 | 9,726 |
| Full stack minus junction | 91 | 99.50% | 0.9967 | 0.9937 | 1,061 | 8,960 |
| **Full stack (all 9)** | **103** | **99.74%** | **0.9986** | **0.9972** | **555** | **4,547** |

### Error Reduction: Base Scores Only -> Full Stack

| Metric | Base only | Full stack | Change | Reduction |
|--------|-----------|-----------|--------|-----------|
| Total FN | 1,468 | 555 | -913 | **-62%** |
| Donor FN | 705 | 257 | -448 | -64% |
| Acceptor FN | 763 | 298 | -465 | -61% |
| Total FP | 14,032 | 4,547 | -9,485 | **-68%** |
| Donor FP | 5,893 | 1,548 | -4,345 | -74% |
| Acceptor FP | 8,139 | 2,999 | -5,140 | -63% |

### What Each Modality Group Contributes

**Conservation alone** (9 features, added to base scores):
- FN: 1,468 -> 974 (**-34%**).  Evolutionary constraint identifies real
  splice sites that the base model underscores.
- FP: 14,032 -> 9,726 (**-31%**).  Non-conserved positions are less likely
  to be functional splice sites, filtering out false predictions.

**Epigenetic + RBP + chromatin** (adding 32 features beyond conservation):
- FP: 9,726 -> 8,960 (**-8%** incremental).  Histone marks and chromatin
  accessibility provide modest additional FP filtering.
- FN: 974 -> 1,061 (+9%).  Slight FN increase — these features add noise
  without junction evidence to anchor them.

**Junction reads** (adding 12 features on top of everything else):
- FN: 1,061 -> 555 (**-48%**).  Direct RNA-seq evidence rescues nearly
  half of the remaining false negatives.
- FP: 8,960 -> 4,547 (**-49%**).  Junction non-support is strong negative
  evidence, cutting false positives in half.

### Key Takeaways

1. **Junction reads are the single most impactful modality** — adding them
   cuts both FN and FP by ~50%.  This is `junction_has_support` being
   the #3 feature by importance (24.8% gain).

2. **Conservation is the second most impactful** — PhyloP/PhastCons add
   34% FN reduction and 31% FP reduction with only 9 features.  This is
   consistent with the known biology: functional splice sites are under
   purifying selection.

3. **The meta-layer does NOT introduce net FPs** — this was a concern.
   Every modality addition reduces FPs.  The full stack has 68% fewer FPs
   than base scores alone.  Multimodal evidence is discriminative, not noisy.

4. **FN rescue is substantial** — 62% of the base model's false negatives
   are recovered.  These are real splice sites that the sequence-only
   model missed but that conservation, junction support, and epigenetic
   marks correctly identify.

5. **Acceptor sites benefit more from FP reduction** (63% reduction) while
   **donor sites benefit more from FN rescue** (64% reduction).  This
   likely reflects the asymmetry in splice site recognition: the GT
   dinucleotide (donor) is more distinctive than the AG dinucleotide
   (acceptor), so the base model already has fewer donor FPs but more
   donor FNs at weak sites.

---

## Top 25 Feature Importances (Full Stack)

| Rank | Feature | Gain | Modality |
|------|---------|------|----------|
| 1 | type_signal_difference | 0.4023 | base_scores |
| 2 | acceptor_prob | 0.2981 | base_scores |
| 3 | junction_has_support | 0.2484 | junction |
| 4 | donor_prob | 0.0246 | base_scores |
| 5 | splice_neither_diff | 0.0119 | base_scores |
| 6 | splice_probability | 0.0036 | base_scores |
| 7 | splice_neither_logodds | 0.0028 | base_scores |
| 8 | neither_prob | 0.0010 | base_scores |
| 9 | donor_acceptor_diff | 0.0007 | base_scores |
| 10 | donor_acceptor_logodds | 0.0007 | base_scores |
| 11 | acceptor_weighted_context | 0.0003 | base_scores |
| 12 | donor_signal_strength | 0.0003 | base_scores |
| 13 | score_difference_ratio | 0.0003 | base_scores |
| 14 | acceptor_signal_strength | 0.0003 | base_scores |
| 15 | junction_tissue_variance | 0.0002 | junction |
| 16 | probability_entropy | 0.0002 | base_scores |
| 17 | distance_to_gene_start | 0.0002 | genomic |
| 18 | relative_gene_position | 0.0002 | genomic |
| 19 | distance_to_gene_end | 0.0002 | genomic |
| 20 | h3k36me3_context_mean | 0.0001 | epigenetic |
| 21 | donor_weighted_context | 0.0001 | base_scores |
| 22 | junction_tissue_mean | 0.0001 | junction |
| 23 | phastcons_score | 0.0001 | conservation |
| 24 | donor_diff_m1 | 0.0001 | base_scores |
| 25 | context_max | 0.0001 | base_scores |

**Caveat**: XGBoost gain importance underestimates the contribution of
conservation and epigenetic features (see SHAP analysis from chr19-22
run in session notes 2026-03-19).  Gain measures how much variance a
feature explains *given the tree structure*; SHAP measures counterfactual
impact.  Conservation features were 10x more important by SHAP than by gain.

---

## Comparison with Previous Baseline

| Metric | chr19-22 baseline | Full-genome |
|--------|-------------------|-------------|
| Train positions | ~466K | 3,851K |
| Test positions | ~96K | 1,993K |
| Features | 83 | 103 |
| Accuracy | 99.78% | 99.74% |
| PR-AUC (donor) | 0.999 | 0.9986 |
| PR-AUC (acceptor) | 0.998 | 0.9972 |
| Split | chr19 train / chr21-22 test | SpliceAI chromosome holdout |

Accuracy is slightly lower on the full genome (99.74% vs 99.78%), which is
expected: the test set is 20x larger and covers more diverse genomic regions.
The consistency validates that the model generalizes across chromosomes.

---

## Implications for M2/M3

The M1 task (canonical splice site classification) is near-saturated — the
base model alone achieves 99.22% and the full stack pushes to 99.74%.  The
more telling signal is in the *error reduction pattern*:

- **Junction reads** are the strongest single predictor beyond base scores.
  For M3 (novel site prediction), junction becomes the *target* — the model
  must predict junction support without it.  The 48% FN reduction from
  junction tells us how much information M3 needs to recover from other
  modalities.

- **Conservation** provides 34% FN reduction independently.  This is the
  primary feature M3 can use as a substitute for junction evidence.  If a
  position is strongly conserved but has no annotation, it may be a novel
  splice site.

- **FP reduction is the main clinical value** for alternative site evaluation (Eval-Ensembl-Alt).
  Base scores alone produce 14K FPs on 2M test positions.  The meta-layer
  cuts this to 4.5K — fewer false calls mean higher clinical specificity.

---

## Reproducibility

All outputs saved to `output/meta_layer/m1_fullgenome/`:

| File | Contents |
|------|----------|
| `xgb_m1_baseline.ubj` | Trained XGBoost model |
| `metrics.json` | Accuracy, PR-AUC, confusion matrix, top features |
| `gene_split.json` | Exact train/val/test gene assignments |
| `ablation_comparison.json` | All four ablation configurations |

To reproduce:
```bash
python examples/meta_layer/01_xgboost_baseline.py \
    --output-dir output/meta_layer/m1_fullgenome
```

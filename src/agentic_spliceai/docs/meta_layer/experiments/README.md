# Meta-Layer Experiments

This directory contains documentation for experiments conducted on the meta-layer architecture for splice site prediction.

---

## Experiment Index

| ID | Name | Status | Outcome | Date |
|----|------|--------|---------|------|
| [001](./001_canonical_classification/) | Canonical Classification | ✅ Completed | Partial Success | Dec 2025 |
| [002](./002_delta_prediction/) | Paired Delta Prediction | ✅ Completed | r=0.38 (insufficient) | Dec 2025 |
| [003](./003_binary_classification/) | Binary Classification (Multi-Step Step 1) | ✅ Completed | AUC=0.61, F1=0.53 | Dec 2025 |
| [004](./004_validated_delta/) | **Validated Delta (Single-Pass)** | ✅ Completed | **r=0.41 (best!)** | Dec 2025 |

---

## Experiment Categories

### Classification-Based Approaches

- **001_canonical_classification**: Train on GTF labels, evaluate on SpliceVarDB (FAILED for variants)
- **003_binary_classification**: Multi-Step Step 1 - "Is this variant splice-altering?"

### Delta-Based Approaches

- **002_delta_prediction**: Paired (Siamese) prediction (r=0.38)
- **004_validated_delta**: **Single-pass with validated targets (r=0.41) - BEST**

---

## Directory Structure

Each experiment follows this structure:

```
NNN_experiment_name/
├── README.md           # Overview, hypothesis, setup, results summary
├── RESULTS.md          # Detailed numerical results (optional)
├── ANALYSIS.md         # In-depth analysis (optional)
├── LESSONS_LEARNED.md  # Key insights and recommendations (optional)
└── (optional)
    ├── config.yaml     # Experiment configuration
    └── figures/        # Plots and visualizations
```

---

## Key Metrics

### For Classification Experiments
- **Accuracy**: Overall classification accuracy
- **AP (Average Precision)**: Per-class ranking quality
- **PR-AUC**: Area under precision-recall curve

### For Delta Prediction Experiments
- **Pearson r**: Correlation with true deltas
- **Detection Rate**: % of splice-altering variants detected
- **Mean |Δ|**: Average absolute delta score

---

## Quick Reference

### Current Best Results

| Task | Best Model | Metric | Value |
|------|------------|--------|-------|
| Classification | Meta-Layer (001) | Accuracy | 99.11% |
| Variant Detection | Validated Delta (004) | Correlation | **r=0.41** |
| Binary Classification | Multi-Step (003) | AUC | 0.61 |

### Key Findings

1. **Classification ≠ Detection**: High classification accuracy doesn't translate to variant detection
2. **Training objective matters**: Must train for the evaluation task
3. **Target quality matters**: Learning from potentially wrong base model deltas limits paired prediction
4. **Validated targets work better**: SpliceVarDB filtering improves correlation from r=0.38 to r=0.41
5. **Binary classification is learnable**: AUC=0.61 > random, but F1=0.53 needs improvement (>0.7)

---

## How to Add a New Experiment

1. Create directory: `NNN_experiment_name/`
2. Copy template from existing experiment
3. Update `README.md` with hypothesis and setup
4. Run experiment, record results
5. Analyze and document insights
6. Update this index

---

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Meta-layer architecture
- [LABELING_STRATEGY.md](../LABELING_STRATEGY.md) - Labeling approaches
- [methods/](../methods/) - Methodology documentation













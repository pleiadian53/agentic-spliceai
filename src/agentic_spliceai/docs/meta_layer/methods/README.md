# Meta-Layer Methodology Documentation

This directory contains documentation for the various methodological approaches being developed for alternative splice site prediction.

## Document Index

| Document | Description | Status |
|----------|-------------|--------|
| [ROADMAP.md](ROADMAP.md) | High-level methodology development roadmap | Active |
| [PAIRED_DELTA_PREDICTION.md](PAIRED_DELTA_PREDICTION.md) | Siamese/paired delta prediction | Tested (r=0.38) |
| [VALIDATED_DELTA_PREDICTION.md](VALIDATED_DELTA_PREDICTION.md) | Single-pass with ground truth targets | **BEST (r=0.41)** |
| [../MULTI_STEP_FRAMEWORK.md](../MULTI_STEP_FRAMEWORK.md) | Decomposed classification approach | In Progress |

## Quick Reference

### Method Summary

```
PAIRED DELTA PREDICTION           VALIDATED DELTA PREDICTION
─────────────────────────         ─────────────────────────
ref_seq ──→ encoder ──┐           alt_seq ──→ encoder ──┐
                      ├─→ diff    ref_base ──→ embed ──┼─→ delta
alt_seq ──→ encoder ──┘           alt_base ──→ embed ──┘

Target: base_delta                Target: validated_delta
Status: r=0.38                    Status: r=0.41 (BEST!)


MULTI-STEP FRAMEWORK
─────────────────────
Step 1: Is splice-altering? → Binary (AUC=0.61, needs >0.7)
Step 2: What type?          → Multi-class (NOT IMPLEMENTED)
Step 3: Where?              → Localization (NOT IMPLEMENTED)
Step 4: How strong?         → Regression (NOT IMPLEMENTED)
```

### Key Differences

| Aspect | Paired Delta | Validated Delta | Multi-Step |
|--------|--------------|-----------------|------------|
| Input | ref + alt | alt + var_info | alt + var_info |
| Target | base_delta | SpliceVarDB-validated | classification |
| Forward passes | 2 | 1 | 1-4 |
| Interpretability | Low | Medium | High |
| Current result | r=0.38 | **r=0.41** | AUC=0.61 |

## Current Status

- **Paired Delta**: Tested, moderate correlation (r=0.38)
- **Validated Delta**: Tested, **best correlation (r=0.41)**
- **Multi-Step Step 1**: Tested, needs improvement (F1=0.53)

## Priority

1. **HIGH**: Scale Validated Delta with more data and HyenaDNA
2. **MEDIUM**: Improve Multi-Step Step 1 (F1 > 0.7)
3. **MEDIUM**: Test Multi-Step Steps 2-4
4. **LOW**: Compare ensemble approaches

## Naming Convention

We use descriptive names instead of cryptic labels:

| Old Name | New Name | Description |
|----------|----------|-------------|
| Approach A | Paired Delta Prediction | Uses both ref and alt sequences |
| Approach B | Validated Delta Prediction | Uses SpliceVarDB-validated targets |
| Phase 1 | Canonical Classification | Training on canonical sites |
| Phase 2 | Delta Prediction | Predicting score changes |

## Related Documentation

- `../experiments/` - Detailed experiment logs
- `../LABELING_STRATEGY.md` - Label derivation strategies
- `../ARCHITECTURE.md` - Model architectures













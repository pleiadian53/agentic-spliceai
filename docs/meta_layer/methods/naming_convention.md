# Meta-Layer Naming Convention

Definitive guide to model and evaluation protocol naming in the
Agentic-SpliceAI meta-layer.  All documents and code should follow
these conventions.

---

## Principle: Models vs Evaluation Protocols

**Models** are what you train — defined by architecture + training labels.
**Evaluation protocols** are how you test — defined by the test set and
the question being asked.

Any model can be evaluated with any protocol.  A model's name should
not encode the evaluation setting.

---

## Models

Models follow the pattern: **M{task}-{level}**

- **Task number** (1-4): the prediction task, from easiest to hardest
- **Level**: S (sequence-level CNN) or P (position-level, e.g. XGBoost)

| Model | Architecture | Training labels | Purpose |
|-------|-------------|----------------|---------|
| **M1-S** | Seq-level dilated CNN | MANE (~370K sites) | Canonical splice classification |
| **M1-P** | XGBoost (position) | MANE | Position-level baseline |
| **M2-S** | Seq-level dilated CNN | Ensembl (~2.8M sites) | Alternative splice site detection |
| **M3-S** | Seq-level dilated CNN | Ensembl, junction=target | Novel site discovery |
| **M4-S** | Seq-level dilated CNN | Variant pairs (planned) | Perturbation-induced splice changes |

### Key distinctions

- **M1-S vs M2-S**: Same architecture, different training labels.  M1-S
  sees only MANE canonical transcripts; M2-S sees the full Ensembl
  annotation including alternative splice sites.
- **M2-S is NOT "M1-S retrained on Ensembl"** — it's a distinct model
  designed for a different task (alternative site detection vs canonical
  classification).
- **M3-S** differs from M2-S in that junction features become the
  **target** (held out) rather than input, forcing the model to predict
  novel sites without RNA-seq evidence.

### Version suffixes (optional)

When architecture changes are significant, append a version:
- `M1-S v1` — probability-space blend (retired)
- `M1-S v2` — logit-space blend with learned temperature (current)

---

## Evaluation Protocols

Evaluation protocols follow the pattern: **Eval-{test_set}**

| Protocol | Test set | Question answered |
|----------|----------|-------------------|
| **Eval-MANE** | MANE splice sites on test chroms | How well does the model classify canonical sites? |
| **Eval-Ensembl-Alt** | Ensembl \ MANE (set difference) | Can the model detect alternative sites beyond MANE? |
| **Eval-GENCODE-Alt** | GENCODE \ MANE (set difference) | Broader alternative site evaluation (curated) |
| **Eval-ClinVar** | ClinVar splice variants | Can delta scores distinguish pathogenic from benign? |
| **Eval-SpliceVarDB** | SpliceVarDB validated variants | Cross-validation against experimental evidence |

### Combining models and protocols

Results are described as: **{Model} on {Protocol}**

Examples:
- "M1-S on Eval-MANE" → canonical classification (PR-AUC 0.9996)
- "M1-S on Eval-Ensembl-Alt" → testing M1-S OOD generalization
- "M2-S on Eval-Ensembl-Alt" → testing M2-S on its target task (PR-AUC 0.965)
- "M2-S on Eval-MANE" → does M2-S maintain canonical performance?

---

## Legacy Naming (Deprecated)

The following names appeared in earlier documents and should be
translated to the current convention:

| Old name | New name | Notes |
|----------|----------|-------|
| M2a | **Eval-Ensembl-Alt** (protocol) | Was ambiguously used as both eval and model |
| M2b | **Eval-GENCODE-Alt** (protocol) | Same ambiguity |
| M2c | **M2-S** (model) | The Ensembl-trained model, not an eval variant |
| M2d | M2-S with junction weighting | Training variant, not a separate model code |
| M2e | Tissue-conditioned M2-S | Future extension |
| M1-S/MANE | **M1-S** | Redundant — M1-S is always MANE-trained |
| M1-S/Ensembl | **M2-S** | This IS the M2 model |
| M2c model | **M2-S** | Clearest name |

---

## File and Directory Naming

### Model outputs
```
output/meta_layer/
  m1s/                  ← M1-S checkpoint (current: v2 logit blend)
  m1s_v1_prob_blend/    ← M1-S v1 (preserved, retired)
  m2s/                  ← M2-S checkpoint (was: m2c/)
```

### Gene caches (annotation-indexed, model-agnostic)
```
gene_cache_mane/        ← MANE train/val/test
gene_cache_ensembl/     ← Ensembl train/val/test
gene_cache_gencode/     ← GENCODE test
```

### Evaluation results
```
output/meta_layer/
  m1s_eval/                     ← M1-S on Eval-MANE
  m2s_eval_ensembl_alt/         ← M2-S on Eval-Ensembl-Alt
  m2s_eval_gencode_alt/         ← M2-S on Eval-GENCODE-Alt
  m1s_eval_ensembl_alt/         ← M1-S on Eval-Ensembl-Alt (OOD test)
```

---

## Summary

```
Models:      M1-S, M2-S, M3-S, M4-S  (what you train)
Protocols:   Eval-MANE, Eval-Ensembl-Alt, Eval-GENCODE-Alt, Eval-ClinVar  (how you test)
Results:     "{Model} on {Protocol}"  (unambiguous)
```

This separation ensures that:
1. Model names encode the **training task**, not the evaluation setting
2. Evaluation protocols are **reusable** across models
3. New models or protocols can be added without renaming existing ones
4. Results are always attributable to a specific model + protocol pair

# Stage 2 — Base Scoring

**Pipeline position:** `data prep → ` **this stage** ` → features → training → eval`

The meta layer is a **refiner**: it improves on a foundation model's splice predictions rather than
predicting from scratch. So before any features are built, the base model runs across the genome and
its raw per-nucleotide probabilities are saved for reuse. For this project the base model is
**OpenSpliceAI** (PyTorch, GRCh38/MANE, ~5× faster than SpliceAI on Apple Silicon).

!!! abstract "Inputs → Outputs"
    **Reads:** reference FASTA + gene windows (registry-resolved).
    **Writes:** `data/<source>/<build>/openspliceai_eval/precomputed/predictions_{chrom}.parquet` —
    per-nucleotide `P(donor)`, `P(acceptor)`, `P(neither)`.

---

## Two ways to get base scores

**Option A — let the feature workflow generate them (simplest).**
[Stage 3](03_feature_engineering.md)'s workflow auto-predicts any chromosome whose
`predictions_{chrom}.parquet` is missing, one chromosome at a time. If you're running the full
pipeline top-to-bottom, you can skip ahead and let Stage 3 pull base scores as needed.

**Option B — pre-generate explicitly (recommended for genome-scale).**
Running the base model once, up front, decouples the slow inference step from feature iteration:

```bash
# Chunked genome-scale prediction that persists raw scores for the meta layer
agentic-spliceai-predict --base-model openspliceai --chunk-size 500 --chromosomes 22
```

The chunked workflow writes raw predictions into the base model's `precomputed/` directory so every
later stage reads cached scores instead of re-running inference. Use `--resume` to continue an
interrupted run.

---

## The one M2-S wrinkle: same model, different windows

This is the subtlety that catches people: **M2-S does not use a different base model.** It still uses
OpenSpliceAI (trained on MANE). What changes is the **region scored** — M2-S scores the base model
over the **Ensembl** gene windows so its base scores line up with the Ensembl labels from
[Stage 1](01_data_preparation.md).

Concretely, that produces a second precomputed directory:

| Model | Base scores directory |
|-------|-----------------------|
| M1-S | `data/mane/GRCh38/openspliceai_eval/precomputed/` |
| M2-S | `data/ensembl/GRCh38/openspliceai_eval/precomputed/` |

Downstream, the training and evaluation scripts point at the right one with `--base-scores-dir`
(see [Stage 4](04_training_m1s.md) / [Stage 5](05_training_m2s.md)).

---

## What's in a prediction parquet

One row per genomic position within the scored gene windows, carrying the three-class base
probabilities. These raw scores are the seed for the 43 engineered `base_scores` features in
[Stage 3](03_feature_engineering.md) (context scores, gradients, peak flags, entropy, cross-type
comparisons) and are also fed to the meta model directly as its base-probability stream.

!!! tip "Report base-model performance the right way"
    When you sanity-check base scores against the Stage 1 labels, evaluate on splice data as the
    imbalanced problem it is — PR-AUC and top-k accuracy at matched recall, not precision/recall at a
    fixed 0.5 threshold. The meta model's whole job is measured as *improvement over this base*, so a
    clean base-vs-meta comparison ([Stage 6](06_evaluation.md)) starts with an honest base number.

---

→ **Next: [Stage 3 — Feature Engineering](03_feature_engineering.md)**

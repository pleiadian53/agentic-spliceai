# Stage 10 — Training M3 (Recognizer + Candidate Refiner)

**Pipeline position:** `09 label curation → ` **this stage** ` → 11 evaluation`

M3 has **two** trainable formulations of the novel-site task, and they have very different footprints:

| Model | What it is | Trainer | Compute |
|-------|-----------|---------|---------|
| **M3-S** (recognizer) | 3-class sequence CNN, junction dropped, annotated sites masked | `07_train_sequence_model.py --mode m3` | **GPU pod** (dense `.npz` cache, like M1-S/M2-S) |
| **M3-R** (candidate refiner) | XGBoost real-vs-artifact over base-matched candidates | `14_train_candidate_refiner.py` | **local** (tabular, no bigWig) |

Read [Stage 4 (M1-S training)](04_training_m1s.md) first for the shared trainer mechanics; below is only
what M3 adds. The formulation rationale is
[methods §4–5](../../meta_layer/methods/06_m3_novel_site_formulation.md#4-recognizer-formulation-m3-s).

---

## M3-S — the recognizer (`--mode m3`)

Same trainer, backbone, and schedule as M1-S/M2-S, with three M3-specific behaviors that `--mode m3`
switches on automatically:

- **junction dropped from inputs** (`mm_channels = 7`) — it is the label, not a feature;
- **annotated sites masked** — label `255`, excluded from the loss (`ignore_index`);
- **confidence weighting** — long-read-confirmed positives up-weighted via `--confirmed-weight`.

```bash
python examples/meta_layer/07_train_sequence_model.py \
    --mode m3 --device cuda \
    --use-shards --epochs 50 --patience 10 --samples-per-epoch 100000 \
    --remove-paralogs --confirmed-weight 2.0 \
    --bigwig-cache /runpod-volume/bigwig_cache \
    --output-dir output/meta_layer/m3_v1
```

`--mode m3` loads `data/mane/GRCh38/m3_labels/{positives_pooled,annotation_mask}.parquet`
([Stage 9](09_m3_label_curation.md)) and builds the 3-class per-gene-window labels. The long pole is the
dense feature cache (bigWig streaming for ~12K train genes) — this is why M3-S is a **pod** job; see the
[GPU Pods runbook](08_gpu_pods.md). Output layout is identical to M1-S
([Stage 4 artifacts](04_training_m1s.md#artifacts-produced)).

!!! note "Tier 1 — the confirmed-only variant"
    `--confirmed-only` retrains on the 77,879 long-read-confirmed positives only (dropping the 76,234
    unconfirmed), to test whether that label noise was the ceiling. It is **not** — `m3s_v1_1_confirmed`
    is marginally worse than `m3_v1`
    ([results](../../meta_layer/results/m3_novel.md#tier-1-does-cleaner-labels-help-no)). The flag exists
    so the experiment is reproducible; the promoted-for-research model remains `m3_v1`.

---

## Folding disease anchors into training (opt-in)

M3-v1 has a quiet failure mode. The curated disease cryptic sites in `disease_anchors.parquet`
([Stage 9, Step 2](09_m3_label_curation.md#step-2-disease-anchors-held-out-d2)) are held out of the
positive pool, so at train time they fall through to class 2 ("neither"), which teaches the recognizer
they are *not* splice sites (the opposite of what they are). An opt-in path fixes this by folding the
`is_novel` subset of the anchors into training instead. Default `off` reproduces M3-v1 exactly.

| Flag | Default | Effect |
|------|---------|--------|
| `--disease-anchors {off,mask,positivize}` | `off` | `off` = M3-v1; `mask` = add anchors to the annotation mask (`255`, a correctness fix that learns nothing); `positivize` = the selected sources become up-weighted positives, the rest masked |
| `--anchor-weight FLOAT` | `5.0` | raw per-anchor loss weight; **ignored** when `--anchor-budget-frac` is set |
| `--anchor-budget-frac FLOAT` | *(unset)* | target fraction (0–1) of total positive-loss mass the positivized anchors should occupy; derives the per-anchor weight from the live positive-mass sum (the durable policy knob) |
| `--anchor-positivize-sources STR` | `sf3b1_cryptic,encode_kd_cryptic` | which `source` values become positives; the rest (`tdp43_cryptic`, the STMN2/UNC13A ALS panel) stay masked so they remain honest held-out probes |

```bash
python examples/meta_layer/07_train_sequence_model.py \
    --mode m3 --device cuda \
    --use-shards --epochs 50 --patience 10 --samples-per-epoch 100000 \
    --remove-paralogs --bigwig-cache /runpod-volume/bigwig_cache \
    --disease-anchors positivize --anchor-budget-frac 0.05 \
    --output-dir output/meta_layer/m3s_anchorpos
```

`fold_disease_anchors()` (in `splice_engine/meta_layer/data/labels.py`) applies the policy, and
`build_m3_labels()` honours an optional per-row `weight` column. `ops_train_m3_pod.sh` runs
`--disease-anchors positivize --anchor-budget-frac 0.05` by default.

!!! note "Anti-circularity is by construction, not by flag"
    Training builds dense caches only for **train-chromosome** genes, so an anchor that lands on a
    held-out (test-chromosome) gene is simply never rasterized even when it is placed in the positive
    class. The held-out D2 truth stays held out; positivizing anchors cannot leak into the
    [anti-circular eval](11_m3_evaluation.md). Keeping `tdp43_cryptic` masked (not positivized) also
    leaves the ALS panel as an untouched cross-mechanism probe.

### Governing the anchor weight

Do **not** tune a raw multiplier. The right question is not "what number" but "what fraction of the
training signal should curated disease biology occupy?" A flat `--anchor-weight 5` works out to only
~1.2% of positive-loss mass at the current corpus (560 anchors against ~232K positive mass); a
policy-consistent **5% budget** derives a per-anchor weight of ~21.8×, and **10%** ≈ 46×. That is
exactly what `--anchor-budget-frac` sets: the budget, not the multiplier.

The durable weighting scheme (only the budget scalar is implemented so far; the rest is design):

```
weight_i = budget × mechanism_allocation_i × evidence_tier_i × [optional site_quality_i]
```

| Component | What it is | How it is set |
|-----------|-----------|---------------|
| `budget` | one scalar, the target positive-loss share (5%, cap 15%) | **set by judgment**, never learned |
| `mechanism_allocation` | `sqrt(n)` or a hard per-mechanism cap, so data volume ≠ importance | parameter-free |
| `evidence_tier` | A = 1.0 / B ≈ 0.6 / C ≈ 0.2 / D = 0 (mask-only) | **measured** from long-read confirmation rate by tier as the anchor set grows (estimated from provenance, not backprop) |
| `site_quality` | optional per-site reliability | dropped initially (belongs as a model feature, not a static label weight) |

!!! warning "Set values, learn facts"
    Learn parameters about the **world** (how reliable each evidence tier is); *set* parameters about
    **values** (how much disease biology may shape the model). Optimizing the budget against the eval
    would just restore mechanism domination and blow the budget. At 563 anchors across 3 mechanisms and
    4 tiers, priors beat fitting.

The GPU-pod bootstrap now self-heals (it repairs a dead resolver's DNS and sets
`PIP_BREAK_SYSTEM_PACKAGES`), so a fresh CA-MTL-1 pod's `pip install` no longer fails; see the
[GPU Pods runbook](08_gpu_pods.md) (`ops/gpu_runner.py` `build_setup_lines`).

---

## M3-R — the candidate refiner (local)

M3-R is the reframe: base model proposes candidates, an XGBoost classifier reranks each *real cryptic vs
artifact*. It is entirely local because the peak-preserving feature parquets already hold every
candidate's multimodal vector. Two steps — build the labels (Stage 9, Step 5), then train:

```bash
# 1. base-score-matched candidate table (Stage 9, Step 5)
python examples/data_preparation/m3/11_build_candidate_labels.py --confirmed-only --neg-ratio 3

# 2. train + diagnose
python examples/meta_layer/14_train_candidate_refiner.py            # full: train + SHAP + ablation
python examples/meta_layer/14_train_candidate_refiner.py --diagnostic-only   # just the go/no-go probe
```

`14` uses the leakage-safe **gene-level SpliceAI split** (test = chr 1/3/5/7/9, the same held-out set as
the recognizer eval), drops junction and the raw base probabilities from the features, and reports the
held-out AUC plus **SHAP-by-modality** and a **leave-one-modality-out** ablation. Output →
`output/meta_layer/m3r_candidate_refiner/` (`m3r_xgb.ubj`, `features.json`, `metrics.json`).

!!! warning "The training AUC is not the verdict"
    M3-R reaches a strong held-out **AUC 0.90 (56% non-base SHAP)** — but that is on the training label
    distribution. Whether it actually helps is decided by the *anti-circular* eval in
    [Stage 11](11_m3_evaluation.md), which is where M3-R's honest-negative result comes from. Do not
    promote on the training AUC.

---

## Which to run

| You want… | Run |
|-----------|-----|
| The best novel-site ranker (research deliverable) | **M3-S** `m3_v1` (pod) |
| To fold curated disease anchors in as positives (SF3B1 / ENCODE-KD) | **M3-S** `--disease-anchors positivize --anchor-budget-frac 0.05` (pod) → `m3s_anchorpos` |
| A cheap, local test of "does multimodal discriminate real cryptic sites?" | **M3-R** `14 --diagnostic-only` |
| To reproduce the Tier 1 label experiment | M3-S `--confirmed-only` (pod) |

---

→ **Next: [Stage 11 — M3 Evaluation](11_m3_evaluation.md)**

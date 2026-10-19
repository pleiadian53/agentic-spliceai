# M4 Tutorial 3: Training and Evaluation

*Part 3 of 4. Previous: [2. Model architecture](02_model_architecture.md). Next: [4. TabPFN](04_tabpfn_in_context_learning.md).*

The question M4 must answer is not "does the model fit the knockdowns it was trained on?" but "can it predict a regulator it has **never seen**?" This part explains how the benchmark is constructed to ask exactly that, how each method is trained, how predictions are scored, and where the results stand.

---

## 1. The question as an experiment

Picture the data as a large, mostly empty matrix: rows are events, columns are regulators, and a cell holds $\Delta$PSI when that knockdown measured that event. Evaluating M4 means hiding some **columns** (regulators) and some **rows** (genes), then predicting the hidden cells.

Two kinds of hiding matter, and both have pitfalls specific to this data.

## 2. Hiding genes: the gene split

Genes are split by chromosome, following the SpliceAI convention: chromosomes 1, 3, 5, 7, and 9 are **test**; the rest are training, with 10% of training genes held aside as validation (`assign_gene_split` in `_m4_common.py`). That gives 4,332 training, 481 validation, and 1,827 test genes.

Splitting by gene keeps every representation of an event on one side: its long and short rows (which are the same measurement with opposite signs), all its regulators, and all its alternative partners. No event appears in two splits.

## 3. Hiding regulators: control-batch-grouped folds

The 185 regulators are divided into five folds of 37; each fold is held out in turn.

The folds must keep **control batches** together. ENCODE ran knockdowns in batches against shared control libraries, so two regulators from one batch have the same control measurement inside their labels:

$$
\Delta\mathrm{PSI}_{e,R} = \mathrm{PSI}^{\text{KD}}_{e,R} - \mathrm{PSI}^{\text{ctrl}}_{e},
\qquad
\Delta\mathrm{PSI}_{e,R'} = \mathrm{PSI}^{\text{KD}}_{e,R'} - \mathrm{PSI}^{\text{ctrl}}_{e}.
$$

If $R'$ is in training and $R$ is held out, the model can partly "predict" $R$ by recalling $R'$, through noise they share rather than biology they share. The 27 batches are inferred from identical control read counts (`control_batch` in the event table), and `regulator_folds(..., groups=...)` assigns whole batches to folds. The first benchmark split batches across folds, and its event prior read 0.742 direction accuracy; with batches kept whole it reads 0.669.

## 4. Two settings

Every method is scored on the same rows: **the held-out regulator's events in test genes**. The settings differ in what is known about those test genes.

| Setting | Known about test genes | Real-world analogue |
|---|---|---|
| **S3** | nothing: no knockdown has been measured on them | a new regulator, on genes never profiled |
| **S2** | their responses to the *training* regulators | a new regulator, on targets profiled under other knockdowns |

S3 is the headline. S2 asks whether knowing how an event behaves under other knockdowns helps predict a new one.

## 5. Few-shot: k examples of the held-out regulator

For few-shot methods, $k$ events of the held-out regulator are revealed, drawn uniformly at random from its events in **training genes**, so no test gene is ever seen under that regulator. The benchmark uses $k \in \{16, 64, 256\}$ and three random draws per $k$, averaging the draws within each regulator.

## 6. How each method is trained

| Method | Training data | Notes |
|---|---|---|
| `zero` | none | predicts 0 |
| `volatility` (S2 only) | training regulators' $\Delta$PSI on the test events | the event's mean response |
| `pooled` | training regulators × training genes (S3), or × all genes (S2) | XGBoost: 400 trees, depth 6, learning rate 0.05, row and column subsampling 0.8; at most 400,000 rows, sampled uniformly; event weights from part 1 |
| `xgb` | the $k$ shots of the held-out regulator | XGBoost: 100 trees, depth 3; unweighted |
| `tabpfn` | the $k$ shots, as an in-context set | no gradient steps (part 4) |
| `pooled_tabpfn` | the $k$ shots' residuals from `pooled` | final prediction = `pooled` + TabPFN correction |

All methods use the same 229-feature compact $X$, and a guard aborts the run if any outcome column (part 1, section 5) is among the features.

## 7. Scoring

Each held-out regulator is a separate task. Metrics are computed **per regulator**, then averaged over regulators, so a regulator with thousands of events does not drown out one with hundreds.

| Metric | Computed on | Definition |
|---|---|---|
| Direction accuracy | significant events | share of events where $\operatorname{sign}(\hat y) = \operatorname{sign}(y)$; a prediction of exactly 0 counts as 0.5 |
| Sign AUROC | significant events | probability that a random increase is ranked above a random decrease |
| Spearman $\rho$ | all events | rank correlation of $\hat y$ with $y$ |
| Weighted MAE | all events | $\sum_i w_i \lvert \hat y_i - y_i \rvert / \sum_i w_i$ with the event weights |

Direction is scored only on significant events (FDR < 0.05, $\lvert y \rvert \ge 0.1$, ≥ 20 reads per condition), because the sign of a tiny, noisy change is not a meaningful target. Chance is 0.5 for both direction metrics. Magnitude is always reported next to direction: a method can get the sign right while predicting changes that are far too large or too small.

## 8. Results

From the v1 benchmark (2026-10-19; full tables in [`m4_fewshot_benchmark.md`](m4_fewshot_benchmark.md)):

| Setting | Method | Direction acc. | Spearman |
|---|---|---:|---:|
| S3 | `pooled`, $k = 0$ | 0.520 | 0.017 |
| S3 | `xgb`, $k = 16 / 64 / 256$ | 0.516 / 0.525 / 0.529 | ≤ 0.017 |
| S3 | `tabpfn`, $k = 16 / 64 / 256$ | 0.515 / 0.534 / **0.565** | **0.032** at $k = 256$ |
| S2 | `volatility` | **0.669** | 0.074 |
| S2 | `pooled`, $k = 0$ | 0.631 | **0.097** |
| S3 rows | rule: $-$(out-of-batch baseline PSI $-$ mean) | 0.624 | 0.050 |

Four readings:

1. **Zero-shot on new genes is near chance** (0.520) once shared control noise is removed; the first run's 0.580 was mostly that noise.
2. **TabPFN learns from a few hundred examples where XGBoost cannot**, reaching above 0.75 for core 3′ splice-site factors (U2AF1, U2AF2, SF3A3).
3. **An event's response to other knockdowns predicts a new one** (0.669 in S2).
4. **The bound rule (0.624) beats every learned S3 method**, which points to the bounded formulation of part 2.

## 9. Running it

```bash
# laptop: labels and regulator features
python examples/data_preparation/m4/01_ingest_kd_dpsi.py
python examples/data_preparation/m4/04_build_regulator_features.py

# GPU pod (bigWig cache on the volume): site features, then both benchmark settings
bash examples/meta_layer/ops_m4_pod.sh

# or the benchmark alone, on any machine with site_features.parquet
python examples/meta_layer/17_m4_fewshot_benchmark.py --setting S3
python examples/meta_layer/17_m4_fewshot_benchmark.py --setting S2 --k 16 64 256

# with TabPFN on a pod: run its ~5,000 in-context fits on 8 worker processes
python examples/meta_layer/17_m4_fewshot_benchmark.py --setting S3 --tabpfn-workers 8
```

Outputs, per setting, in `output/meta_layer/m4_fewshot/<setting>_compact/`: `per_regulator.parquet` (one row per fold, regulator, method, $k$, and draw), `summary.csv` (macro averages), `learning_curves.png`, and `config.json`.

---

## Summary

- Hold out **genes** by chromosome and **regulators** by control-batch-grouped folds; score every method on the same held-out-regulator, held-out-gene rows.
- S3 (nothing known about test genes) is the headline; S2 adds the test events' responses to other knockdowns.
- Direction is scored on significant events, magnitude on all events, both averaged per regulator.

**Next:** [Part 4, TabPFN and in-context learning](04_tabpfn_in_context_learning.md).

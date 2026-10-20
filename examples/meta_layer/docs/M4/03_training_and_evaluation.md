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

## 4. The settings: what is known when predicting

Two things can be new at prediction time: the **regulator** (has M4 seen this knockdown before?) and the **genes** (has M4 seen these events respond to *any* knockdown?). Crossing them gives four settings, named S0 to S3:

| | Genes seen (events profiled under other knockdowns) | Genes unseen (no knockdown measured on them) |
|---|---|---|
| **Regulator seen** | S0: in-sample fit, not a test | **S1**: a known regulator on new targets |
| **Regulator unseen** | **S2**: a new regulator on profiled targets | **S3**: a new regulator on new targets |

The benchmark runs **S2 and S3**, because M4's purpose is to predict regulators it has not seen. S1 (predict a known regulator's effect on new genes) is a useful but easier question and is not implemented yet. There is no S4.

In both S2 and S3, every method is scored on the same rows: **the held-out regulator's events in test genes**. The settings differ only in what the methods may use about those test genes.

### S3: new regulator, new genes (the headline)

- **What is withheld:** the held-out regulator entirely, and every knockdown measurement on test genes.
- **What a method may use:** the training regulators' responses on *training* genes, the features of the test events (sequence, base scores, conservation, chromatin, GTEx usage, binding), and optionally $k$ examples of the held-out regulator from training genes.
- **Real-world analogue:** a regulator nobody has profiled, acting on genes nobody has profiled under any knockdown.
- **Goal:** test whether M4 has learned something transferable about how regulators act on events, rather than memorising responses.

### S2: new regulator, profiled genes

- **What is withheld:** the held-out regulator only.
- **What a method may use:** everything in S3, plus how the *test* events responded to the training regulators. That enables the `volatility` prior (an event's average response to other knockdowns) and lets the pooled model train on test genes under other regulators.
- **Real-world analogue:** a lab that has already profiled a panel of knockdowns in its cell line and adds one new regulator.
- **Goal:** test how much an event's behaviour under other knockdowns predicts a new one, and whether anything regulator-specific adds to it.

Comparing the two settings separates two sources of signal: *what kind of event this is* (available in both) and *how this event has behaved before* (S2 only).

## 5. k: examples of the held-out regulator

$k$ is the number of **labelled events of the held-out regulator** a method is allowed to see.

- $k = 0$ is **zero-shot**: the method knows nothing about the new regulator except, where available, its eCLIP binding features.
- $k = 16, 64, 256$ are **few-shot**: that many of the regulator's events, with their measured $\Delta$PSI, are revealed. They are drawn at random from its events in **training genes**, so no test gene is ever seen under that regulator. Each $k$ is drawn three times and the results averaged within each regulator.

Only methods that can use examples have results at $k > 0$: `xgb` (trained on the $k$ examples alone), `tabpfn` (reads them as context), and `pooled_tabpfn` (corrects the pooled model with them). Methods at $k = 0$ (`zero`, `volatility`, `pooled`) are shown as single reference values.

### Reading a results table

```text
S3: unseen regulator, unseen genes (direction accuracy on significant events; chance 0.5)

Method                              k=0     k=16    k=64    k=256
pooled model                        0.520
XGBoost on the regulator's examples         0.516   0.525   0.529
TabPFN v2                                   0.515   0.534   0.565
rule, no learning (bound effect)    0.624
```

- Each cell is **direction accuracy**: among the held-out regulator's *significant* test events, the share where the predicted sign of $\Delta$PSI matches the measured sign; averaged over the 185 regulators. A coin flip scores 0.5.
- Reading across a row is a **learning curve**: how much a method gains as it sees more of the new regulator.
- Reading down a column compares methods with the **same information**.
- The *rule* row is a reference with no learning: it predicts that events above average control PSI will fall and those below will rise. It knows nothing about the regulator, so any learned method should beat it; none does yet in S3.

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

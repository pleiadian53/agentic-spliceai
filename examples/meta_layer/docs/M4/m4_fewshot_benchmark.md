# M4 benchmark v0: predicting a new regulator's splicing response

**Status:** first end-to-end benchmark, 2026-10-17; **corrected 2026-10-18** for a shared-control leak (section 3.1). Development-stage results. The corrected S2 event prior is final; the model rows (`pooled`, `xgb`) are v0 numbers awaiting a rerun under the corrected folds and features, and TabPFN methods are not yet run. Artifact: `output/meta_layer/m4_fewshot/` (registered, tag `m4`).

M4 asks a conditional question that M1–M3 cannot: *if we knock down this RNA-binding protein, which splicing choices change, and in which direction?* This document records the first benchmark that answers it with held-out regulators, explains how the benchmark was built so that its gains are real, and states what the first numbers do and do not show.

---

## 1. The task in one paragraph

An **A3SS** event is a choice between two competing acceptors spliced to the same upstream donor; an **A5SS** event is the mirror image, two competing donors spliced to one downstream acceptor. We call the farther-reaching choice the *long* form. For an event $e$ and a knocked-down regulator $R$ (an RNA-binding protein), rMATS measures the fraction of transcripts using the long form, the **percent spliced in** (PSI), in knockdown and in control samples. The target is

$$
\Delta\mathrm{PSI}_{e,R} \;=\; \mathrm{PSI}^{\text{KD}}_{e,R} - \mathrm{PSI}^{\text{ctrl}}_{e,R} \;\in\; [-1, 1],
$$

the signed change in usage of the long form when $R$ is depleted. Inputs must be available **before** the knockdown outcome is observed: sequence, base-model scores, conservation, chromatin, GTEx junction usage, and where $R$ binds. A method is scored on regulators it never trained on, optionally given $k$ labeled events of that regulator ("$k$ shots").

**Scope.** 185 ENCODE shRNA knockdowns in **HepG2**, A3SS/A5SS events only. This is a benchmark of alternative-site responses to RBP depletion, not of cryptic-exon discovery or disease mutations.

## 2. The data

| Artifact | Size | What it holds |
|---|---|---|
| `kd_events.parquet` | 1,585,467 event × regulator rows | full event geometry (long, short, and flanking exon; the two competing sites and their shared **partner** site), read counts, $\Delta$PSI |
| `site_features.parquet` | 41,127 sites × 118 cols | the meta-layer feature vector of every long, short, and partner site, all nine modalities |
| `regulator_site_features.parquet` | 4,007,346 site × regulator rows | where the knocked-down regulator itself binds (eCLIP), oriented to the intron or exon side |

Three properties of the corpus shape everything below:

1. **The event, not the site, is the unit.** The same competing-site pair measured against a different flanking exon is a different splicing choice; within one knockdown, 48% of such pairs differ by more than 0.1 in $\Delta$PSI. Events are keyed by full geometry (`event_key`).
2. **The matrix is partially observed.** rMATS reports an event only where it has reads, so an event is measured under a median of **53 of 185** knockdowns. Missingness is part of the problem.
3. **Signal is thin and noisy.** Control read support per event has quartiles 8 / 26 / 81 (10th, 25th, 50th percentile). Only **1.49%** of rows (23,690) pass the prespecified significance rule below, split evenly between increases (12,027) and decreases (11,663).

The panel is HepG2 for all 185 knockdowns. The source tables carry no cell-line column; this was verified from control-sample read support, which is about 1,000-fold higher in liver marker genes (ALB, APOB, FGB, TF, SERPINA1) than in erythroid/K562 markers (HBG1/2, GATA1, GYPA).

## 3. Building the benchmark so that gains are real

Most of the work went into closing paths by which a model could look good without learning regulation. Each row below was measured, not assumed.

| Trap | Evidence | Resolution |
|---|---|---|
| The long and short forms are algebraically coupled | $\Delta\mathrm{PSI}_\text{long} = -\Delta\mathrm{PSI}_\text{short}$ for all 1,585,467 events | one target per event (the long form); never split the pair |
| rMATS event IDs are per file | the same event has a different `event_uid` under every regulator | split by **gene** (SpliceAI chromosomes 1/3/5/7/9 held out); 0 events span two splits |
| Repeated competing-site pairs | 88,298 (regulator, pair) groups span >1 flanking exon | keep them as distinct events; weight so each (regulator, pair) cluster sums to 1 |
| The knockdown's own control PSI shares noise with the label | corr($\Delta$PSI, own control PSI) = **−0.247**, vs **−0.047** for a baseline built from the *other* knockdowns' controls | never an input; use the leave-this-experiment-out baseline PSI |
| Low-coverage events | 25% of events have ≤ 26 control reads | significance requires FDR < 0.05, $\lvert\Delta\mathrm{PSI}\rvert \ge 0.1$, **and** ≥ 20 reads per condition |
| **Shared control samples** (section 3.1) | 93.1% of rows share control counts with another knockdown; 27 control batches | group regulator folds by control batch; baseline PSI leaves out the whole batch |
| Feature-store selection bias | the genome-wide feature store covers only 65% of these sites (it is score-thresholded), and its base-score context features read sampled, not adjacent, neighbours | compute every site from contiguous base predictions; no site dropped |

The second-to-last trap deserves a sentence of intuition. $\Delta\mathrm{PSI} = \mathrm{PSI}^\text{KD} - \mathrm{PSI}^\text{ctrl}$, so if sampling noise happens to push a control PSI up, the same noise pushes $\Delta\mathrm{PSI}$ down. A model given its own experiment's control PSI learns that regression to the mean and looks predictive while knowing nothing about the regulator.

### 3.1 The trap found after v0: shared controls

ENCODE ran its knockdowns in batches against shared control libraries. Two knockdowns in one batch report *identical* control junction counts for an event: 93.1% of event × regulator rows match another knockdown's controls, and an event measured under a median of 53 knockdowns has only 14 distinct control profiles. Regulator pairs either share controls on essentially every event they both measure or on none (0 of 17,020 pairs in between), which defines **27 control batches** (median 6 regulators, largest 17, 10 singletons).

Why it matters: for two regulators $R$ and $R'$ in one batch,

$$
\Delta\mathrm{PSI}_{e,R} = \mathrm{PSI}^{\text{KD}}_{e,R} - \mathrm{PSI}^{\text{ctrl}}_{e},
\qquad
\Delta\mathrm{PSI}_{e,R'} = \mathrm{PSI}^{\text{KD}}_{e,R'} - \mathrm{PSI}^{\text{ctrl}}_{e},
$$

with the *same* control term. Its sampling noise enters both labels with the same sign, so one regulator's response "predicts" the other's for reasons unrelated to regulation. v0 split batch-mates across folds, which leaked this noise in two places:

- **The S2 event prior.** Averaging other regulators' $\Delta$PSI included batch-mates. Corrected (batch-grouped folds), its direction accuracy falls from **0.742 to 0.669** and Spearman from 0.208 to 0.074.
- **The baseline-PSI feature.** Leaving out only the row kept batch-mates' copies of the same control. Correlation with $\Delta$PSI: own control −0.247, row-out baseline −0.047, batch-out baseline −0.025 (the remainder is plausibly real: a site near full usage cannot rise much).

Both are fixed in code: `kd_events.parquet` carries a `control_batch` column, regulator folds keep batches whole, and `baseline_psi_loo` leaves out the batch.

## 4. Settings and methods

Regulators are split into five folds, **keeping each control batch in one fold** (since 2026-10-18); each fold is held out in turn. Every method is scored on the **same rows**: the held-out regulator's events in held-out genes. Shots are drawn only from the regulator's events in training genes.

| Setting | What is known about the test genes | Question |
|---|---|---|
| **S3** | nothing: no knockdown response of any test-gene event | can we predict a new regulator on new genes? |
| **S2** | their responses under the *other* (training) regulators | a new assay on targets we have profiled before |

| Method | $k$ | What it is |
|---|---|---|
| `zero` | – | predict 0: the magnitude floor |
| `volatility` | – | S2 only: the event's mean $\Delta$PSI under the training regulators |
| `pooled` | 0 | XGBoost trained on all training regulators (site, geometry, and regulator-binding features) |
| `xgb` | 16, 64, 256 | XGBoost trained on the held-out regulator's $k$ shots alone |
| `tabpfn`, `pooled_tabpfn` | $k$ | TabPFN in context on the shots, alone or on the pooled residual (**pending license**) |

Features (229, "compact" set): event geometry (alternative-site distance, junction spans, exon lengths), the partner site's feature vector, long − short feature differences, the leave-one-out baseline PSI, the number of other knockdowns observing the event, and the regulator's own binding near each of the three sites.

Metrics are computed per regulator and then averaged over regulators, since each regulator is one task: **direction accuracy** on significant events (a prediction of exactly 0 counts as a coin flip), **sign AUROC** on significant events, **Spearman** $\rho$ on all events, and event-weighted **MAE**.

## 5. Results

### S3: new regulator, new genes

v0 numbers. Test genes share no events with training genes, so the fold leak does not reach S3 directly; the baseline-PSI feature carried a small amount of shared control noise (section 3.1), so `pooled` may be slightly optimistic until the rerun.

| Method | Direction acc. | Sign AUROC | Spearman | MAE (weighted) | MAE (significant) |
|---|---:|---:|---:|---:|---:|
| zero | 0.500 | 0.500 | – | **0.0932** | 0.2304 |
| **pooled** ($k$=0) | **0.580** | **0.588** | **0.069** | 0.0943 | **0.2274** |
| xgb, $k$=16 | 0.516 | 0.500 | 0.005 | 0.1331 | 0.2401 |
| xgb, $k$=64 | 0.527 | 0.511 | 0.011 | 0.1186 | 0.2316 |
| xgb, $k$=256 | 0.532 | 0.518 | 0.019 | 0.1078 | 0.2290 |

Learning curves for each run are written to `output/meta_layer/m4_fewshot/<setting>_compact/learning_curves.png` (images outside `docs/` are not tracked in git).

### S2: new regulator, previously profiled genes

| Method | Direction acc. | Sign AUROC | Spearman | MAE (weighted) | Status |
|---|---:|---:|---:|---:|---|
| zero | 0.500 | 0.500 | – | 0.0932 | final |
| **volatility, batch-grouped folds** | **0.669** | **0.703** | **0.074** | 0.0950 | **final (corrected)** |
| volatility, v0 folds | 0.742 | 0.795 | 0.208 | 0.0898 | leaky, superseded |
| pooled ($k$=0), v0 folds | 0.695 | 0.742 | 0.169 | 0.0905 | leaky, rerun pending |
| xgb, $k$=256 | 0.532 | 0.518 | 0.019 | 0.1078 | v0 (fits only the regulator's own shots) |

The v0 S2 figure is superseded and not shown; it will be regenerated with the rerun.

All values are macro averages over 185 regulators (three shot draws per $k$, averaged within a regulator first). The per-regulator standard deviation of direction accuracy is about 0.12, and a regulator has a median of only 26 significant test events, so single-regulator numbers are noisy; the averages are not.

### Per-regulator view

- **Regulator-specific gains over the event prior: withdrawn pending the rerun.** v0 reported the pooled model beating the prior for 36 of 185 regulators (QKI +0.10, U2AF1 +0.08). Both sides of that comparison used leaky folds, so it is not reported as a finding until it is recomputed.
- **The best zero-shot (S3) regulators** with ≥ 30 significant test events include CPSF6 (0.74), EIF4A3 (0.73), TIAL1 (0.73), QKI (0.72), HNRNPU (0.71), and U2AF1 (0.68).
- **The regulator's own eCLIP binding does not help yet.** Regulators with HepG2 eCLIP score the same as regulators with none (S3 direction accuracy 0.574 vs 0.581). Only 1.6% of (site, regulator) pairs with eCLIP have the regulator bound within ±300 nt, so binding is too sparse to carry most responses, which are presumably indirect.

## 6. What the results mean

**An event's response to other knockdowns is the strongest single predictor, but weaker than v0 suggested.** Averaging an event's response under knockdowns from *other* control batches gives 67% direction accuracy (chance 50%), above the zero-shot feature model on new genes (58%). Some events are volatile and tend to move the same way whatever is depleted. v0's 74% overstated this because batch-mates share control noise; how much regulator-specific signal sits on top of the prior is the open question the rerun answers.

**Learning a regulator from its own handful of examples fails at these sizes.** With 229 features and at most 256 shots, a per-regulator XGBoost overfits: its error is worse than predicting zero at every $k$. This is precisely the small-data regime that in-context tabular models (TabPFN) and shrinkage toward a strong prior are designed for, which is why they are the next methods to run.

**S2 is a cold-start problem.** The event × regulator matrix is partially observed and dominated by a shared response axis. A new regulator's $k$ shots are best used to *place* it relative to known regulators, as in collaborative filtering, rather than to fit a model from scratch.

## 7. Limitations

- One cell line (HepG2) and two event types (A3SS/A5SS). The TDP-43 cryptic exons in UNC13A and STMN2 are skipped-exon events and are outside this corpus.
- The epigenetic and chromatin features use the current ENCODE track versions; four of the older genome-wide feature store's tracks predate an accession update, so this table and that store differ on H3K36me3 and DNase (correlation 0.87 to 0.99).
- Pooled-model hyperparameters are untuned, and pooled training rows are subsampled to 400,000.
- The direction metric is defined only on significant events (1.49% of rows); Spearman over all events is dominated by near-zero changes.

## 8. Next steps

0. **Rerun S3 and S2** under batch-grouped folds and the batch-out baseline (the pod driver runs both settings in about ten minutes); regenerate the figures and the per-regulator comparison.
1. **TabPFN** in context on the shots, alone and on the pooled residual (`pooled_tabpfn`).
2. **Cold-start factorisation for S2**: low-rank completion of the event × regulator matrix; fit a new regulator's loading from its $k$ shots.
3. **Residual stacking**: volatility prior + pooled model + $k$-shot correction, evaluated as a learning curve.
4. **Regulator descriptors for S3**: motif and binding-profile summaries that let the pooled model tell regulators apart without shots.
5. **Strata**: directly bound vs unbound events; MANE vs Ensembl-only genes.

## 9. Reproduce

```bash
# data (laptop): event table + regulator binding features
python examples/data_preparation/m4/01_ingest_kd_dpsi.py
python examples/data_preparation/m4/04_build_regulator_features.py

# site features (GPU pod with the bigWig cache): all modalities, checkpointed per chromosome
python examples/data_preparation/m4/03_build_site_features.py \
    --bigwig-cache /runpod-volume/bigwig_cache --resume

# benchmark (either machine)
python examples/meta_layer/17_m4_fewshot_benchmark.py --setting S3
python examples/meta_layer/17_m4_fewshot_benchmark.py --setting S2
```

`examples/meta_layer/ops_m4_pod.sh` runs the pod stages end to end, with a preflight that fails if any required input or bigWig track is missing.

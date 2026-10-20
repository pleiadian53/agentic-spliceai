# M4 benchmark: predicting a new regulator's splicing response

**Status:** v1, 2026-10-19. The first run (v0, 2026-10-17) leaked shared control noise between regulators (section 3.1); v1 reruns every method with control-batch-grouped folds and an out-of-batch baseline feature, and adds TabPFN v2. Development-stage results. Artifact: `output/meta_layer/m4_fewshot/` (registered, tag `m4`).

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
| `tabpfn`, `pooled_tabpfn` | 16, 64, 256 | TabPFN v2 in context on the shots, alone or on the pooled model's residuals (S3 only; `tabpfn` depends only on the shots, so it would be identical in S2) |

Features (229, "compact" set): event geometry (alternative-site distance, junction spans, exon lengths), the partner site's feature vector, long − short feature differences, the out-of-batch baseline PSI, the number of other knockdowns observing the event, and the regulator's own binding near each of the three sites.

Metrics are computed per regulator and then averaged over regulators, since each regulator is one task: **direction accuracy** on significant events (a prediction of exactly 0 counts as a coin flip), **sign AUROC** on significant events, **Spearman** $\rho$ on all events, and event-weighted **MAE**.

## 5. Results

All values are macro averages over 185 regulators (three shot draws per $k$, averaged within a regulator first). A regulator has a median of only 26 significant test events, so single-regulator numbers are noisy (standard deviation of direction accuracy about 0.12); the averages are not. Learning curves are written to `output/meta_layer/m4_fewshot/<setting>_compact/learning_curves.png` (images outside `docs/` are not tracked in git).

### S3: new regulator, new genes

| Method | Direction acc. | Sign AUROC | Spearman | MAE (weighted) |
|---|---:|---:|---:|---:|
| zero | 0.500 | 0.500 | – | **0.093** |
| pooled ($k$=0) | 0.520 | 0.519 | 0.017 | 0.096 |
| xgb, $k$ = 16 / 64 / 256 | 0.516 / 0.525 / 0.529 | 0.500 / 0.511 / 0.515 | 0.005 / 0.010 / 0.017 | 0.133 / 0.119 / 0.108 |
| **tabpfn**, $k$ = 16 / 64 / 256 | 0.515 / 0.534 / **0.565** | 0.503 / 0.518 / **0.555** | 0.004 / 0.014 / **0.032** | 0.123 / 0.101 / 0.095 |
| pooled_tabpfn, $k$ = 16 / 64 / 256 | 0.518 / 0.526 / 0.545 | 0.505 / 0.512 / 0.536 | 0.011 / 0.017 / 0.025 | 0.124 / 0.103 / 0.098 |
| *reference rule, no learning:* $-(\text{out-of-batch baseline PSI} - \text{mean})$ | *0.624* | *0.639* | *0.050* | – |

### S2: new regulator, previously profiled genes

| Method | Direction acc. | Sign AUROC | Spearman | MAE (weighted) |
|---|---:|---:|---:|---:|
| zero | 0.500 | 0.500 | – | 0.093 |
| **volatility** | **0.669** | **0.703** | 0.074 | 0.095 |
| **pooled** ($k$=0) | 0.631 | 0.670 | **0.097** | **0.092** |
| xgb, $k$=256 | 0.529 | 0.515 | 0.017 | 0.108 |

### What changed from v0

| Number | v0 (leaky) | v1 | Why |
|---|---:|---:|---|
| S3 pooled direction acc. | 0.580 | 0.520 | v0's baseline-PSI feature left out only the row, keeping batch-mates' copies of the same control |
| S2 event prior direction acc. | 0.742 | 0.669 | v0 split control batches across folds |
| S2 pooled direction acc. | 0.695 | 0.631 | both of the above |

Most of v0's apparent zero-shot signal was shared control noise.

### Per-regulator view

- **TabPFN learns the spliceosome's 3′ splice-site machinery.** At $k = 256$, TabPFN beats XGBoost for 122 of 185 regulators. Its best regulators (≥ 30 significant test events) are QKI (0.813), U2AF1 (0.811), RBM25 (0.764), MAGOH (0.762), U2AF2 (0.756), and SF3A3 (0.752). U2AF1, U2AF2, and SF3A3 recognise the 3′ splice site and branch point, exactly the decision an A3SS event makes; QKI has a strong sequence motif; MAGOH belongs to the exon-junction complex. Regulators with strong, coherent programs are learnable from a few hundred examples.
- **Regulator-specific signal on top of the event prior is real but concentrated.** In S2 the pooled model beats the event prior for 53 of 185 regulators, most clearly NONO (+0.12), RBM25 (+0.11), SF3A3 (+0.11), IGF2BP2 (+0.09), and SRSF3 (+0.09).
- **The regulator's own eCLIP binding helps only marginally.** TabPFN at $k = 256$ scores 0.577 for regulators with HepG2 eCLIP, 0.551 with K562-only eCLIP, and 0.560 with none. Only 1.6% of (site, regulator) pairs with eCLIP have the regulator bound within ±300 nt, so most responses are presumably indirect.

## 6. What the results mean

**Zero-shot prediction on new genes is not yet solved.** With the leak removed, the pooled model is near chance on unseen regulators and unseen genes (0.520). Nothing in the current features tells the model *which* regulator was knocked down beyond its sparse binding.

**A few hundred examples of a regulator help, if the learner has a prior.** TabPFN improves steadily with $k$ and overtakes XGBoost from $k = 64$; at $k = 256$ it reaches 0.565 overall and above 0.75 for core splicing factors. XGBoost trained on the same shots overfits. Stacking TabPFN on the pooled model (`pooled_tabpfn`) does not help, because the pooled prior itself carries little signal in S3.

**The bound structure of PSI is the largest untapped signal.** A rule with no learning, predicting that events near full usage fall and events near zero rise, scores 0.624 in S3, above every trained method. The pooled model has the out-of-batch baseline as a feature but does not exploit it: squared-error regression on targets that are almost all near zero learns to predict tiny changes. Building the bound into the model (perturbed PSI through a log-odds offset from an independent baseline, [tutorial part 2](02_model_architecture.md)) is the most direct next step.

**An event's behaviour under other knockdowns predicts a new one.** On profiled genes, the event prior reaches 0.669, and the pooled model (which sees those responses through training) adds regulator-specific gains for about a quarter of regulators. This is the cold-start structure that matrix factorisation is built for.

## 7. Limitations

- One cell line (HepG2) and two event types (A3SS/A5SS). The TDP-43 cryptic exons in UNC13A and STMN2 are skipped-exon events and are outside this corpus.
- The epigenetic and chromatin features use the current ENCODE track versions; four of the older genome-wide feature store's tracks predate an accession update, so this table and that store differ on H3K36me3 and DNase (correlation 0.87 to 0.99).
- Pooled-model hyperparameters are untuned, and pooled training rows are subsampled to 400,000.
- The direction metric is defined only on significant events (1.49% of rows); Spearman over all events is dominated by near-zero changes.
- `pooled_tabpfn` was run in S3 only; an S2 run was stopped by a crashed worker and not repeated, since the pooled prior it builds on carries little S3 signal.

## 8. Next steps

1. **Bounded formulation:** predict perturbed PSI as $\sigma(\operatorname{logit}(p_0) + \delta)$ with an independent $p_0$ and a beta-binomial loss on knockdown read counts, so the bound effect (worth 0.624 on its own) is built in.
2. **Cold-start factorisation for S2:** low-rank completion of the event × regulator matrix; fit a new regulator's loading from its $k$ shots, and compare with TabPFN at the same $k$.
3. **Regulator descriptors for S3:** motif and binding-profile summaries, or protein language-model embeddings, so the model can tell regulators apart without shots.
4. **Strata:** directly bound vs unbound events; MANE vs Ensembl-only genes.

## 9. Reproduce

```bash
# data (laptop): event table + regulator binding features
python examples/data_preparation/m4/01_ingest_kd_dpsi.py
python examples/data_preparation/m4/04_build_regulator_features.py

# site features (GPU pod with the bigWig cache): all modalities, checkpointed per chromosome
python examples/data_preparation/m4/03_build_site_features.py \
    --bigwig-cache /runpod-volume/bigwig_cache --resume

# benchmark (either machine)
python examples/meta_layer/m4/01_m4_fewshot_benchmark.py --setting S3
python examples/meta_layer/m4/01_m4_fewshot_benchmark.py --setting S2
```

`examples/meta_layer/m4/ops_m4_pod.sh` runs the pod stages end to end, with a preflight that fails if any required input or bigWig track is missing.

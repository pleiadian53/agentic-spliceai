# M4 Tutorial 1: How the Training Set Is Composed (X and y)

*Part 1 of 4 in the M4 tutorial series. Next: [2. Model architecture](02_model_architecture.md).*

M4 predicts how a splicing choice responds when an RNA-binding protein (RBP) is knocked down. Before looking at any model, it helps to be precise about what one training example is, what the label $y$ measures, and what goes into the feature vector $X$. This part builds that picture from a single real observation outward.

---

## 1. One observation

ENCODE knocked down 185 RBPs in the liver cancer cell line HepG2 and sequenced RNA from each knockdown and from matched controls. rMATS, a differential-splicing caller, compared the two for every **alternative 3′ or 5′ splice-site event** (A3SS / A5SS).

Take one. In gene ZNF706, two competing acceptors (the *long* and *short* choices) are both joined to one upstream donor. Under knockdown of the RBP AATF:

| | Control | Knockdown |
|---|---:|---:|
| Fraction of transcripts using the long acceptor (PSI) | 0.244 | 1.000 |

PSI ("percent spliced in") is the share of transcripts that use the long form. The **label** is its change,

$$
y = \Delta\mathrm{PSI} = \mathrm{PSI}^{\text{KD}} - \mathrm{PSI}^{\text{ctrl}} = 1.000 - 0.244 = +0.756 .
$$

The short form loses exactly what the long form gains, so one number describes the event.

## 2. The unit: one event under one regulator

A training example is a pair $(e, R)$: an event $e$ and the knocked-down regulator $R$. An event is defined by **three sites**:

```text
A3SS (two acceptors compete, one donor shared)        A5SS (two donors compete, one acceptor shared)

[flank exon]D----------A_long[=====alt exon====]      [==alt exon==]D_long--------A[flank exon]
[flank exon]D---------------A_short[==alt exon=]      [=alt ex=]D_short-----------A[flank exon]
            ^ partner                                                             ^ partner
```

Why three and not one? The same pair of competing sites spliced to a different partner is a different choice, and it can move differently. ZNF706 shows this directly: with a second upstream donor, the very same acceptor pair under the same AATF knockdown gives $\Delta\mathrm{PSI} = -0.206$ instead of $+0.756$. Across the corpus, 88,298 (regulator, competing-pair) groups appear with more than one partner, and their $\Delta$PSI differs by more than 0.1 in 48% of cases.

The corpus (`data/mane/GRCh38/m4_labels/kd_events.parquet`, built by `examples/data_preparation/m4/01_ingest_kd_dpsi.py`):

| Quantity | Value |
|---|---|
| Event × regulator rows | 1,585,467 |
| Distinct events (full geometry) | 19,340 |
| Regulators | 185, all HepG2 |
| Regulators observing a typical event | median 53 of 185 |

The last line says the event × regulator matrix is **partially observed**: rMATS reports an event only where that experiment has reads, so most cells are missing. A missing cell is not a zero effect.

## 3. The label y

$$
y_{e,R} = \Delta\mathrm{PSI}_{e,R} = \mathrm{PSI}^{\text{KD}}_{e,R} - \mathrm{PSI}^{\text{ctrl}}_{e,R} \in [-1, 1].
$$

Positive means the long form gained usage under knockdown. Two properties of $y$ shape everything downstream.

**It is noisy.** PSI is a ratio of read counts, and many events have few reads: the 10th, 25th, and 50th percentiles of control reads per event are 8, 26, and 81. The ZNF706 change of $+0.756$ rests on 5 control reads and 8 knockdown reads.

**Most changes are small.** A change counts as *significant* only when

$$
\mathrm{FDR} < 0.05,
\qquad
\lvert\Delta\mathrm{PSI}\rvert \ge 0.1,
\qquad
\text{reads}_{\text{KD}},\ \text{reads}_{\text{ctrl}} \ge 20 ,
$$

which holds for 23,690 rows (1.49%): 12,027 increases and 11,663 decreases. Training uses every row; the significant subset is where direction is scored (part 3). ZNF706's $+0.756$, despite FDR 0.0003, fails the read rule.

## 4. The features X

Every feature must be knowable **before** the knockdown outcome is observed. $X$ is assembled from three sources.

### 4.1 Site features (step 03)

Each of the three sites gets the meta layer's 100-column multimodal vector (`examples/data_preparation/m4/03_build_site_features.py`):

| Modality | Cols | What it says about a site |
|---|---:|---|
| Base scores | 43 | OpenSpliceAI probabilities and their local shape |
| Genomic | 4 | position in the gene, GC content |
| Conservation | 9 | phyloP / phastCons |
| Epigenetic | 12 | H3K36me3 / H3K4me3 across 5 cell lines |
| Junction | 12 | GTEx split-read usage in normal tissues |
| RBP binding (any RBP) | 8 | ENCODE eCLIP peak density |
| Chromatin accessibility | 12 | ATAC (5 lines) + DNase (5 tissues) |

Every site is computed from contiguous per-nucleotide predictions, including sites in genes outside MANE, which are scored on demand. None is dropped.

### 4.2 Regulator binding (step 04)

Where does the knocked-down regulator **itself** bind near each site? From ENCODE eCLIP (`04_build_regulator_features.py`), six numbers per site (peak counts within ±50 / 300 / 1,000 nt, on the intron side, on the exon side, and distance to the nearest peak), plus two flags saying whether eCLIP exists for $R$ and whether it is from HepG2. Of the 185 regulators, 71 have HepG2 eCLIP, 33 only K562, and 81 none (features null).

### 4.3 Event-level features

Geometry (alternative-site distance, two junction spans, three exon lengths, event type), a **baseline PSI**, and the number of other knockdowns that observe the event.

### 4.4 The assembled vector

The default *compact* representation (`examples/meta_layer/_m4_common.py`) has 229 columns:

$$
X_{e,R} = \big[\ \underbrace{g_e}_{7}\ ,\ \underbrace{s_{\text{long}} - s_{\text{short}}}_{100}\ ,\ \underbrace{s_{\text{partner}}}_{100}\ ,\ \underbrace{b_e}_{2}\ ,\ \underbrace{r_{e,R}}_{20}\ \big]
$$

where $g_e$ is geometry, $s$ are site vectors, $b_e$ the baseline block, and $r_{e,R}$ the regulator's binding near the three sites. The long − short difference encodes the competition the label measures.

Notice what is **not** in $X$: the regulator's name. There is no one-hot vector or embedding of $R$. Only $r_{e,R}$ depends on the regulator, and for the 81 regulators without eCLIP it is empty. Part 2 discusses what that implies.

## 5. Columns that must never enter X

| Column | Why |
|---|---|
| `psi_kd`, `fdr`, `pvalue`, knockdown reads | they are, or are computed from, the outcome |
| this row's own control PSI | shares noise with $y$ (below) |

The second is subtle. Since $y = \mathrm{PSI}^{\text{KD}} - \mathrm{PSI}^{\text{ctrl}}$, sampling noise that raises the measured control PSI lowers $y$. A rule with no learning at all, $\hat y = -(\mathrm{PSI}^{\text{ctrl}} - \text{mean})$, scores direction accuracy 0.691 and Spearman 0.297 on held-out data, better than every trained model, purely from this regression to the mean. The baseline PSI in $X$ is therefore built from *other* experiments' controls.

"Other experiments" needs care too. ENCODE ran knockdowns in batches against **shared** control libraries: 93.1% of rows have identical control counts to another knockdown, forming 27 control batches (`control_batch` column). The baseline leaves out the row's whole batch. With that, its correlation with $y$ drops from $-0.247$ (own control) to $-0.025$, and what remains is plausibly real: a site already used 100% of the time can only go down.

## 6. Weights

Each (regulator, competing-site pair) group gets total weight one, so a site pair measured against three partners counts once rather than three times. This is cluster balancing, not a confidence weight.

---

## Summary

| Symbol | Meaning |
|---|---|
| $(e, R)$ | one training example: event $e$ (three sites) under knockdown of regulator $R$ |
| $y$ | signed long-form $\Delta$PSI, in $[-1, 1]$; noisy and usually small |
| $X$ | 229 pre-outcome features: geometry, site differences, partner context, out-of-batch baseline PSI, $R$'s binding |
| weight | 1 / size of the (regulator, competing-pair) group |

**Next:** [Part 2, model architecture](02_model_architecture.md): which models consume this $X$ today, and what a properly conditional M4 would look like.

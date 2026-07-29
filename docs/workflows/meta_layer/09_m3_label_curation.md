# Stage 9 — M3 Label Curation (Novel-Site Sub-Series)

**Pipeline position:** `feature engineering (Stage 3) → ` **M3 sub-series** ` → 10 training → 11 evaluation`

M3 (novel splice sites) reuses Stages 1–3 unchanged — the same data prep, base scoring, and multimodal
feature parquets as M1-S/M2-S. What it does **not** share is its labels: a novel site has no annotation
to learn from, so M3's label set has to be *curated from evidence* (junction reads, long-read isoforms,
disease catalogs). This stage is that curation. It is pure data engineering — no model, no GPU — and it
produces every parquet the M3 recognizer and refiner consume.

!!! abstract "Inputs → Outputs"
    **Reads:** GTEx junctions, SpliceVault, disease catalogs, ENCODE long-read GTFs, the annotation union.
    **Writes:** the label pools under `data/mane/GRCh38/m3_labels/` and the D1 truth under
    `data/encode_longread/GRCh38/`.

The *why* behind each choice — recognizer + post-filter, junction-as-label, base-matched negatives — is
[06_m3_novel_site_formulation.md](../../meta_layer/methods/06_m3_novel_site_formulation.md). This doc is
the *how* and *in what order*.

---

## What gets built

Solid arrows contribute rows; **dashed arrows are exclusions** (subtracted, not added).

```mermaid
flowchart TD
  A["GTEx junctions + SpliceVault<br/><i>short-read evidence</i>"] --> P["<b>positives_pooled.parquet</b><br/>154,113 novel sites<br/>→ M3-S train"]
  C["GENCODE ∪ RefSeq<br/><i>existing annotation</i>"] --> M["<b>annotation_mask.parquet</b><br/>825,746<br/>→ loss-ignore + post-filter"]
  E["ENCODE long-read GTFs<br/><i>full-length reads</i>"] --> D1["<b>longread_truth_novel.parquet</b><br/>681,809<br/>→ EVAL truth D1"]
  B["SF3B1 / ENCODE-KD / TDP-43<br/><i>disease catalogs</i>"] --> D2["<b>disease_anchors.parquet</b><br/>6,351<br/>→ EVAL truth D2"]
  P -- "positives" --> R["<b>candidate_labels.parquet</b><br/>52,320 base-matched<br/>→ M3-R train"]
  M -. "excluded" .-> R
  D1 -. "excluded (no leakage)" .-> R
  D2 -. "excluded (no leakage)" .-> R
  D2 -. "anti-joined out" .-> P
```

### Who consumes what

Two things make this stage's outputs confusing until you see the shape: there are **two M3 models**, and
the artifacts split into **training labels vs evaluation truth** — which must never mix.

- **M3-S**, the *recognizer* — a sequence CNN that scores every position in a gene and ranks candidate
  novel sites ([Stage 10](10_training_m3.md)).
- **M3-R**, the *candidate refiner* — an XGBoost model that reranks base-proposed candidates as
  real-vs-artifact.
- **D1 / D2** are the two **independent** truth sets used at [Stage 11](11_m3_evaluation.md). They are
  named because M3 cannot be evaluated against its own training annotation (that would be circular):
  **D1** = ENCODE long-read confirmed novel junctions, **D2** = held-out disease cryptic sites.

| Artifact | What it is | Used by | When | Built by |
|----------|-----------|---------|------|----------|
| `positives_pooled.parquet` | 154,113 evidence-supported novel sites, with a `longread_confirmed` flag | **M3-S** | **train** (positive class) | `03_ingest_splicevault.py` → `04_merge_positives.py` |
| `negatives.parquet` | GT/AG dinucleotide decoys + easy non-sites | **M3-S** | **train** (decoys; largely vestigial — see [methods §4](../../meta_layer/methods/06_m3_novel_site_formulation.md)) | `09_build_negatives.py` |
| `annotation_mask.parquet` | 825,746 already-annotated sites (GENCODE ∪ RefSeq) | **M3-S** | **dual role** — at train time the loss *ignores* these positions (so a known site is never scored as a negative); at inference they are *subtracted* so only novel calls survive | `09_build_negatives.py` |
| `candidate_labels.parquet` | 52,320 base-proposed candidates labelled real-vs-artifact, with negatives matched to the positives' base-score distribution | **M3-R** | **train** | `11_build_candidate_labels.py` |
| `longread_truth_novel.parquet` | 681,809 novel junctions confirmed by full-length long reads | both | **eval — D1** | `10_build_longread_truth.py` |
| `disease_anchors.parquet` | 6,351 disease cryptic sites (TDP-43 / SF3B1 / ENCODE-KD), anti-joined out of training | both | **eval — D2** | `05`/`06`/`07_ingest_*` → `08_finalize_anchors.py` |

!!! warning "The leakage rule that shapes this whole stage"
    D1 and D2 are **excluded from every training set** — the anchors are anti-joined out of
    `positives_pooled`, and Step 5 excludes `annotation ∪ positives ∪ D1 ∪ D2` when sampling negatives.
    Without that, a held-out truth site could be trained on as an "artifact" and the anti-circular
    evaluation would be meaningless.

Scripts live in
[`examples/data_preparation/m3/`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/data_preparation/m3/).

---

## Step 1 — the positive pool (novel sites)

Novel positives come almost entirely from **SpliceVault** (cryptic events observed across 335K RNA-seq
samples) plus a small **GTEx-novel** survivor set from the cross-annotation audit
(`01_cross_annotation_audit.py`, which intersects GTEx junction sides against GENCODE/RefSeq/UCSC to keep
only the genuinely-unannotated ones).

```bash
python examples/data_preparation/m3/03_ingest_splicevault.py   # SpliceVault → parquet
python examples/data_preparation/m3/04_merge_positives.py      # + GTEx-novel → positives_pooled.parquet
```

`04` also anti-joins the pool against the annotation union (so every positive is genuinely novel) and
stamps the **`longread_confirmed`** flag used later by `--confirmed-weight` / Tier 1.

## Step 2 — disease anchors (held-out D2)

TDP-43, SF3B1, and ENCODE-KD cryptic-site catalogs are ingested and finalized into a **held-out** eval
set — anti-joined out of the training positives so they never leak.

```bash
python examples/data_preparation/m3/05_ingest_sf3b1_anchors.py
python examples/data_preparation/m3/06_ingest_encode_kd_anchors.py
python examples/data_preparation/m3/07_ingest_tdp43_anchors.py
python examples/data_preparation/m3/08_finalize_anchors.py     # → disease_anchors.parquet (is_novel flag)
```

## Step 3 — the mask and the recognizer decoys

```bash
python examples/data_preparation/m3/09_build_negatives.py
```

Writes `annotation_mask.parquet` (GENCODE ∪ RefSeq-curated — the set the loss ignores *and* the
inference post-filter subtracts) and `negatives.parquet` (GT/AG decoys + easy non-sites). Note the
negatives are dinucleotide decoys with ~0 base score — fine as recognizer decoys, but **not** usable for
M3-R, which needs base-matched negatives (Step 5).

## Step 4 — the independent long-read truth (D1)

```bash
python examples/data_preparation/m3/10_build_longread_truth.py
```

Builds `longread_truth_novel.parquet` from ENCODE long-read (PacBio/ONT) transcriptome GTFs — junctions
confirmed by full-transcript reads, absent from annotation. This is the **anti-circular** eval truth:
independent of the SpliceVault short-read signal the positives came from.

## Step 5 — base-matched candidate labels (for M3-R)

```bash
python examples/data_preparation/m3/11_build_candidate_labels.py --confirmed-only --neg-ratio 3
```

Joins the peak-preserving feature parquets to the labels and samples **base-score-matched hard
negatives** — positives = confirmed novel sites in the parquets; negatives = base-proposed positions
*not* in `annotation ∪ positives ∪ D1 ∪ D2` (all eval sets excluded so no held-out truth is trained as
an artifact), stratified to the positives' base-score histogram. Output `candidate_labels.parquet`
(52,320 rows). Fully local — no bigWig, no pod.

---

## Verify

```bash
python -c "import polars as pl; d='data/mane/GRCh38/m3_labels/'; \
print('positives', pl.read_parquet(d+'positives_pooled.parquet').height); \
print('anchors',   pl.read_parquet(d+'disease_anchors.parquet').height); \
print('candidates',pl.read_parquet(d+'candidate_labels.parquet').height)"
```

Expected: 154,113 / 6,351 / 52,320. Coordinate integrity is checked by the GT/AG-by-strand oracle (all
positives sit at a canonical dinucleotide on the correct strand) — see
[`examples/meta_layer/docs/M3/m3_training_data.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/M3/m3_training_data.md).

---

→ **Next: [Stage 10 — Training M3](10_training_m3.md)**

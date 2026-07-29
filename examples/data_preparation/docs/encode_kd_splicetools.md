# ENCODE RBP-Knockdown Splicing Tables (SpliceTools / rMATS)

## Overview

`data/encode_kd_splicetools/1_RBP_kd/` holds **precomputed differential-splicing
tables** for a large panel of RNA-binding-protein (RBP) knockdowns. Each table
answers one question for one regulator:

> *When RBP **R** is knocked down, which splice sites change how often they are
> used, and by how much?*

Concretely it is **rMATS JCEC output** for **185 RBP knockdowns** (each vs its
matched control), split by alternative-splicing event type. This is the project's
only source of **paired, perturbation-attributed ΔPSI** — the measured effect of
losing a specific regulator — as opposed to static annotation or population
mis-splicing. It is consumed by two workflows:

- **M3** ([`../m3/06_ingest_encode_kd_anchors.py`](../m3/06_ingest_encode_kd_anchors.py)) —
  the KD-*gained* sites become held-out disease **anchors** (mechanism-attributed
  cryptic splice sites, a generalization test for the novel-site model).
- **M4** ([`../m4/01_ingest_kd_dpsi.py`](../m4/01_ingest_kd_dpsi.py)) — the full,
  signed, per-regulator ΔPSI becomes the **training corpus** for perturbation-induced
  splicing (`kd_dpsi.parquet`).

---

## Provenance

The ENCODE consortium ran RNA-seq on hundreds of shRNA/CRISPR RBP knockdowns, but
the **ENCODE portal hosts only BAM/bigWig/FASTQ/gene-quantifications — no splicing
tables.** The differential-splicing quantification we use was computed and published
separately by the **Flemington lab** in the **SpliceTools** paper, which knocked
down **186 known RBPs** (splicing factors and others) from ENCODE and ran
**rMATS 4.1.1** to call alternative-splicing changes (A3SS, A5SS, MXE, RI, SE).

| | |
|---|---|
| **Tool / dataset** | SpliceTools (downstream RNA-splicing analysis suite) |
| **Publication** | Yang et al., *SpliceTools…*, **Nucleic Acids Research** 2023, 51(7):e42 · PMID/PMC **PMC10123099** · DOI `10.1093/nar/gkad111` |
| **Underlying RBP–eCLIP context** | Van Nostrand et al., *A large-scale binding and functional map of human RNA-binding proteins*, Nature 2020 (the ENCODE RBP program) |
| **Code / data** | GitHub `flemingtonlab/SpliceTools` · web `splicetools.org` · archived data Zenodo **`10.5281/zenodo.7603628`** |
| **Splicing caller** | rMATS 4.1.1 (`--task both`; JCEC = junction + exon-body counts) |
| **Cell context** | ENCODE knockdown lines. The SpliceTools reference panel is **HepG2**; the project's earlier sourcing notes describe these as K562/HepG2. The tables carry **no cell-line column**, so if cell-type attribution matters for a specific analysis, confirm against the source bundle. |

> This dataset was selected during a survey of candidate catalogs as the best
> **mechanism-attributed** set — each splicing change is attributable to a known
> perturbed regulator, rather than merely correlated with a phenotype.

### How it was acquired (and how to re-acquire)

**There is no fetch script in this repo for this dataset** — it was a **one-time
manual download** of the SpliceTools data bundle into `data/encode_kd_splicetools/`.
The ENCODE portal cannot supply it directly (no rMATS there). To reproduce or
extend the local copy, pull the rMATS tables from the SpliceTools distribution:

```bash
# SpliceTools ships the precomputed rMATS bundle for the 186 ENCODE RBP KDs.
# GitHub: https://github.com/flemingtonlab/SpliceTools   (see data/ + docs)
# Zenodo archive (DOI 10.5281/zenodo.7603628) is the citable snapshot.
# Place the *_test_cntl_<EVENT>.MATS.JCEC.txt tables under:
#   data/encode_kd_splicetools/1_RBP_kd/
```

---

## Directory layout & file naming

```
data/encode_kd_splicetools/
└── 1_RBP_kd/                        # "category 1": the RBP-knockdown panel
    ├── <RBP>_test_cntl_A3SS.MATS.JCEC.txt   # alt 3' splice site (acceptor)
    ├── <RBP>_test_cntl_A5SS.MATS.JCEC.txt   # alt 5' splice site (donor)
    └── …                                    # 185 RBPs × {A3SS, A5SS} = 370 files
```

- **`<RBP>`** = the knocked-down regulator (gene symbol), e.g. `SF3B1`, `U2AF1`,
  `TARDBP`. Parsed in code as `Path(fp).name.split("_test_cntl_")[0]`.
- **`test` = knockdown (SAMPLE_1)**, **`cntl` = control (SAMPLE_2)**.
- **`.MATS.JCEC.txt`** = the rMATS *JCEC* table (reads counted from splice junctions
  **and** exon-body coverage; the alternative `JC` variant uses junction reads only —
  this project uses JCEC).
- **Only `A3SS` and `A5SS` are present locally.** The upstream SpliceTools bundle
  also has `SE` (skipped/cassette exon), `RI` (retained intron), and `MXE`
  (mutually exclusive exons); those were **not** downloaded — see [Scope](#scope--gotchas).

---

## Column reference (rMATS A3SS/A5SS JCEC)

Each row is one alternative-splicing **event** (a pair of competing isoforms) with
the counts and PSI in knockdown vs control.

| Column | Meaning |
|---|---|
| `ID` | rMATS event id (unique within the file) |
| `GeneID` | Ensembl gene id (quote-wrapped in the raw file) |
| `geneSymbol` | Gene symbol (quote-wrapped) |
| `chr`, `strand` | Chromosome (`chr`-prefixed) and strand |
| `longExonStart_0base`, `longExonEnd` | Coordinates of the **long** exon form (0-based start) |
| `shortES`, `shortEE` | Coordinates of the **short** exon form |
| `flankingES`, `flankingEE` | The constitutive flanking exon (shared by both forms) |
| `ID` (2nd) | A duplicated id column; polars renames it `ID_duplicated_0` — ignore it |
| `IJC_SAMPLE_1`, `SJC_SAMPLE_1` | **Inclusion / Skipping** junction+body counts in the **KD** (comma-separated per replicate) |
| `IJC_SAMPLE_2`, `SJC_SAMPLE_2` | Inclusion / Skipping counts in the **control** |
| `IncFormLen`, `SkipFormLen` | Effective lengths used to normalize counts → PSI |
| `PValue`, `FDR` | rMATS significance of the KD-vs-control difference |
| `IncLevel1` | **PSI of the inclusion (long) form in the KD**, per replicate (may contain `NA`) |
| `IncLevel2` | PSI of the inclusion (long) form in the **control**, per replicate |
| `IncLevelDifference` | **ΔPSI = mean(IncLevel1) − mean(IncLevel2) = PSI_KD − PSI_control** |

**PSI** ("percent spliced in") is the fraction of transcripts using the inclusion
(long) isoform: `PSI = (IJC/IncFormLen) / (IJC/IncFormLen + SJC/SkipFormLen)`.

---

## Event geometry — what "long vs short" means

A3SS/A5SS events are **competing splice-site choices**, not exon skipping. The long
and short isoforms **share one exon boundary** and **differ at the regulated one**;
their usage is complementary (using one acceptor precludes the other).

```
A5SS (alternative DONOR, + strand)          A3SS (alternative ACCEPTOR, + strand)
  ┌──────── exon ────────┐···intron···       ···intron···┌──── exon ────────┐
  │              long donor ┘                         └ long acceptor        │
  │        short donor ┘                                └ short acceptor     │
  shared start (flanking side)                    shared end (flanking side)
```

- **A5SS** regulates the **donor** (5′ splice site); **A3SS** regulates the
  **acceptor** (3′ splice site).
- The **regulated** boundary is the one that differs (long vs short); the other is
  shared. `01_ingest_kd_dpsi.py::_BOUNDARY` encodes exactly which coordinate column
  is the regulated site per (event, strand).
- On the **minus strand** the roles of the start/end coordinate columns flip
  (donor ↔ exon end/start), which is why the project always validates positions
  with a **GT/AG dinucleotide oracle split by strand** rather than trusting the raw
  column semantics.

---

## Interpreting ΔPSI (the biology)

`IncLevelDifference` is signed, and the sign carries the mechanism:

| Sign of ΔPSI (KD − control) | Meaning for the long/inclusion site |
|---|---|
| **> 0** | The site is used **more** after knockdown → the RBP normally **represses** it → **de-repression** (a cryptic/aberrant site "switching on"). |
| **< 0** | The site is used **less** after knockdown → the RBP normally **activates/promotes** it. |
| **≈ 0** | The RBP does not regulate this event (most rows). |

Significance is `FDR` (with `PValue`); a common threshold is **FDR < 0.05 and
|ΔPSI| ≥ 0.1**. Across the whole panel, significant events split roughly evenly
between de-repression and repression — both directions are present, which is what
makes this dataset usable for learning the *direction* of a perturbation (not just
detecting a site).

**This is the signal the base/M1/M2/M3 models cannot supply:** those are trained on
a single (normal) cellular state, so they can *detect* a splice site but have never
seen the *perturbed* state that switches it on. The KD tables are the paired WT↔KD
evidence.

---

## How the project uses it

### M3 — held-out cryptic anchors (KD-gained only)
[`../m3/06_ingest_encode_kd_anchors.py`](../m3/06_ingest_encode_kd_anchors.py)
keeps only the **KD-gained** form (`IncLevelDifference > 0` → long form gained; the
gained boundary is the de-repressed cryptic site), filtered to `FDR < 0.01 &
|ΔPSI| ≥ 0.1`, and **collapses all RBPs** to one representative per site. Output:
`data/mane/GRCh38/m3_labels/anchors/anchors_encode_kd.parquet` (~5,745 mechanism-
attributed cryptic sites), reserved as a generalization test — never trained on.

### M4 — perturbation-paired ΔPSI corpus (full, signed, per-regulator)
[`../m4/01_ingest_kd_dpsi.py`](../m4/01_ingest_kd_dpsi.py) keeps the **full signed
distribution**, emits **both** competing sites per event (long `+ΔPSI`, short
`−ΔPSI`), and **never collapses the regulator** — `rbp` is M4's conditioning
variable. Output: `data/mane/GRCh38/m4_labels/kd_dpsi.parquet` (3,170,934 signed
per-site rows over 185 RBPs). See [`../m4/README.md`](../m4/README.md) for the schema
and [`../../meta_layer/docs/M4/m4_conditional_design.md`](../../meta_layer/docs/M4/m4_conditional_design.md)
for the model design.

Both consumers share the same coordinate treatment: the raw rMATS boundary is mapped
to an exonic splice-site position via a **per-(strand, splice_type) GT/AG offset
scan** (absorbs rMATS 0-based conventions), validated to ≥ ~0.98 canonical
dinucleotide by strand.

---

## Scope & gotchas

- **A3SS/A5SS only (locally).** Cassette/poison exons (e.g. the ALS UNC13A and
  STMN2 cryptic exons) are **SE** events, which are **not in the local bundle** —
  they exist upstream in SpliceTools and can be added by downloading the `SE`
  tables. Until then, UNC13A/STMN2 stay covered by the dedicated TDP-43 anchors
  ([`../m3/07_ingest_tdp43_anchors.py`](../m3/07_ingest_tdp43_anchors.py)).
- **Cell context.** Cancer-line knockdowns — the *right* signal for cancer-induced
  splicing, distinct from neuronal contexts. No cell-line column in the tables.
- **`NA` in `IncLevel*`.** A knockdown with a single replicate has no comma and can
  be read as a float column; replicate strings can also contain `NA`. Parsers must
  cast to string and skip `NA` (see `01_ingest_kd_dpsi.py::_mean_psi`).
- **Duplicate `ID` column.** The header has two `ID` columns; the second is a
  duplicate (polars → `ID_duplicated_0`). Use the first.
- **Quote-wrapped `GeneID`/`geneSymbol`.** Read with `quote_char='"'` (or strip the
  quotes) to dequote.
- **rMATS coordinates are 0-based / mixed** — never use the raw boundary as the
  splice position; run the strand-split GT/AG offset scan.

---

## Related

- Consumers: [`../m3/06_ingest_encode_kd_anchors.py`](../m3/06_ingest_encode_kd_anchors.py) (M3 anchors) · [`../m4/01_ingest_kd_dpsi.py`](../m4/01_ingest_kd_dpsi.py) (M4 corpus)
- Workflows: [`../m3/README.md`](../m3/README.md) · [`../m4/README.md`](../m4/README.md)
- Model design: [`../../meta_layer/docs/M4/m4_conditional_design.md`](../../meta_layer/docs/M4/m4_conditional_design.md)
- Sibling data doc: [`ground_truth_custom_genomes.md`](ground_truth_custom_genomes.md)

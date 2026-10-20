# M4 data preparation — perturbation-paired ΔPSI label workflow

Run-ordered data workflow that produces the **M4 training label corpus**
(perturbation-induced splicing). M4 is the *conditional* sibling of M1/M2/M3:
given a regulator perturbation (knock down RBP *R*), predict the resulting
**signed ΔPSI** at splice sites. The missing ingredient is perturbation-paired
labels — WT vs knockdown ΔPSI across many regulators — which this workflow builds
from ENCODE shRNA-knockdown rMATS tables.

This directory holds *data preparation* only. Model training / evaluation lives
under `examples/meta_layer/`. Design + findings docs live under
[`examples/meta_layer/docs/m4/`](../../meta_layer/docs/m4/).

> Run each step after `mamba activate agentic-spliceai`.

## MLOps position

```
curate → INGEST/PROCESS (this dir) → VALIDATE → train M4 (meta_layer) → eval → monitor → deploy
                                         ▲
                        splice-site strand validator (GT/AG oracle gate)
```

The GT/AG offset scan runs inside step 01 and hard-fails below an 0.85 canonical
rate, so a coordinate/build regression cannot pass silently.

## Preconditions

1. **External download** (one-time, already present): ENCODE-KD SpliceTools rMATS
   A3SS/A5SS JCEC tables → `data/encode_kd_splicetools/1_RBP_kd/` (370 files =
   185 RBPs × {A3SS, A5SS}, ENCODE knockdown lines). Provenance, re-acquisition,
   and full column reference:
   [`../docs/encode_kd_splicetools.md`](../docs/encode_kd_splicetools.md).
2. **Strand-correct FASTA + index**: `data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa`
   (+ `.fai`), for the dinucleotide validation.

## Run order

| # | Script | Produces | Notes |
|---|---|---|---|
| 01 | `01_ingest_kd_dpsi.py` | `data/mane/GRCh38/m4_labels/{kd_events,kd_dpsi}.parquet` | event table (full geometry) + per-site view; GT/AG-validated |
| 02 | `02_characterize_corpus.py` | `output/meta_layer/m4_labels/{corpus_report.md, corpus_stats.json}` | distribution, sign balance, per-RBP, KD∩eCLIP reconciliation |
| 03 | `03_build_site_features.py` | `data/mane/GRCh38/m4_labels/site_features.parquet` + `output/meta_layer/m4_labels/site_features_report.json` | feature vector per event site, computed from contiguous base predictions; no site dropped |
| 04 | `04_build_regulator_features.py` | `data/mane/GRCh38/m4_labels/regulator_site_features.parquet` | the knocked-down RBP's own eCLIP binding near each event site (HepG2 peaks preferred) |

## Final label set (consumed by M4 training)

| Artifact | Count | Role |
|---|---:|---|
| `kd_events.parquet` | 1,585,467 rows | **the modeling grain**: one row per (event, RBP) with full event geometry (19,340 geometries) |
| `kd_dpsi.parquet` | 3,170,934 rows | per-site view of the same events (= 2 × 1,585,467; 27,171 unique regulated sites) |
| `site_features.parquet` (step 03) | 41,127 sites | multimodal feature vector of every long, short, and partner site |
| `regulator_site_features.parquet` (step 04) | 4,007,346 (site, RBP) pairs | binding of the knocked-down RBP itself near each site; null when it has no eCLIP (81 of 185; 71 have HepG2 peaks, 33 K562 only) |

### Event table (`kd_events.parquet`)

A site coordinate alone does not identify the measured splicing choice: an A3SS
event chooses between two acceptors **spliced to one upstream donor** (the flanking
exon), and the same acceptor pair measured against a different flanking exon is a
different choice. Within a single knockdown, 88,298 alternative-site pairs appear with
more than one flanking exon, and their ΔPSI differs by >0.1 in 48% of cases. These are
not duplicates (no two rows share the full geometry), so they are kept, keyed by
geometry, and grouped together at split time.

| Column | Meaning |
|---|---|
| `event_key` | full geometry `event_type:chrom:strand:long_exon:short_exon:flank_exon`, stable across RBPs (rMATS numeric IDs are per file) |
| `altpair_key` | `event_type:chrom:strand:long_site:short_site`, groups events that differ only in flanking exon |
| `long_site`, `short_site` | the competing alternative sites (acceptors for A3SS, donors for A5SS), 1-based exonic base |
| `partner_site` | the shared junction partner on the flanking exon (donor for A3SS, acceptor for A5SS) |
| `*_exon_start/end` | long, short, and flanking exon coordinates, 1-based inclusive |
| `canonical_long/short/partner` | GT/AG check per site (rates 0.976 / 0.977 / 0.992) |
| `dpsi`, `psi_kd`, `psi_ctrl` | long-form ΔPSI (= `IncLevelDifference`) and mean replicate PSIs |
| `ijc_*`, `sjc_*`, `n_rep_*`, `*_form_len` | read support per condition (control reads p10/p25/p50 = 8/26/81) |
| `fdr`, `pvalue`, `rbp`, `rmats_id`, `cell_line` | rMATS statistics, regulator, per-file ID, `HepG2` |

**Split rule.** `dpsi(long) = −dpsi(short)` exactly, so the two forms must never be
separated; and `event_uid` (built from the per-file rMATS ID) differs across RBPs for
the same event, so grouping by it still leaks an event across regulators. The main
benchmark splits by **gene**, which keeps every representation of an event together.

**Baseline PSI.** A knockdown's own `psi_ctrl` shares sampling noise with its ΔPSI
(regression to the mean), so it is not a clean input. Use a leave-this-experiment-out
baseline: the geometry's mean control PSI across the *other* knockdowns.

**Schema** (KEY for feature/annotation joins = `[chrom, position, strand] + splice_type`,
bare chrom, int position, strand-aware — same convention as `m3_labels/`):
`chrom, position, strand, splice_type, event_type, form, dinuc, canonical_dinuc,
rbp, dpsi, inclvl_diff, psi_kd, psi_ctrl, fdr, pvalue, gene_id, gene_symbol,
event_id, event_uid, paired_position, source, build_origin`.

**Framing (Option B — per-regulated-site, signed).** Each A3SS/A5SS event's long
and short isoforms use competing (complementary) alternative sites. Every event
emits **two rows per RBP**: the long-form site with `dpsi = +IncLevelDifference`
and the short-form site with `dpsi = −IncLevelDifference`. This matches M4's
per-splice-site output grain (like M1/M2/M3) and makes "does knocking down *R*
raise or lower usage of *this* donor/acceptor" a direct label. Key properties:

- **Regulator identity (`rbp`) is the conditioning variable — never collapsed.**
  One row per (site, RBP). The (event × regulator) → ΔPSI matrix is **partially
  observed**: rMATS reports an event only where it has read support, so a geometry is
  measured in a median of 53 of the 185 knockdowns (15.8% in all 185).
- **Both ΔPSI signs are kept** (repression and de-repression) — the "both states"
  the design doc names as the missing training signal.
- **No FDR/|dPSI| prefilter** — the full distribution is retained; downstream
  thresholds (`fdr`, `pvalue`, `dpsi` are columns). Event-level significance is
  ~52K at FDR<0.05 / ~24K at FDR<0.01, balanced ~50/50 between the two directions.
- **`event_uid` / `form` / `paired_position`** let training split leakage-safely
  by event (never leaking a long/short pair across train/val) and recover the pair.

## Scope boundaries

- **A3SS/A5SS only.** SE/RI/MXE are essentially absent from this download, so
  cassette/poison exons (UNC13A, STMN2) are **out of scope** here; they remain the
  held-out TDP-43 anchors produced by
  [`../m3/07_ingest_tdp43_anchors.py`](../m3/07_ingest_tdp43_anchors.py).
- **HepG2 (hepatocellular carcinoma line)** — verified from the data: the tables have no
  cell-line column, but control read support in liver markers (ALB, APOB, FGB, TF,
  SERPINA1) exceeds erythroid/K562 markers (HBG1/2, GATA1, GYPA) ~1,000-fold for all
  185 knockdowns. No cell-context conditioning is possible from this corpus. It is the
  right signal for the cancer-induced M4 arm
  (design doc §3); the neuronal/ALS TDP-43 arm is separate and sparse. Note TARDBP
  is one of the 104 regulators with both a KD effect and eCLIP binding.
- **rMATS-native coordinates, GT/AG-validated by strand** (offset scan), not
  liftover.

## Next: M4 training (a `meta_layer` step)

Once characterized, plan regulator-conditioned per-site ΔPSI prediction: choose the
regulator representation (data-level counterfactual ablation — proven in
[`../../UI_integration/04_tdp43_ablation_counterfactual.py`](../../UI_integration/04_tdp43_ablation_counterfactual.py)
— vs a learned RBP embedding), split leakage-safely by `event_uid`/gene, and
validate the **de-repression sign** on held-out SF3B1 and TDP-43 anchors.

## Related

- Design + findings: [`../../meta_layer/docs/m4/`](../../meta_layer/docs/m4/)
  (`m4_conditional_design.md`).
- Sibling label workflow: [`../m3/README.md`](../m3/README.md) (novel splice sites).

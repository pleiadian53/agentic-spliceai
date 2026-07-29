# M4 data preparation — perturbation-paired ΔPSI label workflow

Run-ordered data workflow that produces the **M4 training label corpus**
(perturbation-induced splicing). M4 is the *conditional* sibling of M1/M2/M3:
given a regulator perturbation (knock down RBP *R*), predict the resulting
**signed ΔPSI** at splice sites. The missing ingredient is perturbation-paired
labels — WT vs knockdown ΔPSI across many regulators — which this workflow builds
from ENCODE shRNA-knockdown rMATS tables.

This directory holds *data preparation* only. Model training / evaluation lives
under `examples/meta_layer/`. Design + findings docs live under
[`examples/meta_layer/docs/M4/`](../../meta_layer/docs/M4/).

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
| 01 | `01_ingest_kd_dpsi.py` | `data/mane/GRCh38/m4_labels/kd_dpsi.parquet` | signed per-site ΔPSI, per RBP; GT/AG-validated |
| 02 | `02_characterize_corpus.py` | `output/meta_layer/m4_labels/{corpus_report.md, corpus_stats.json}` | distribution, sign balance, per-RBP, KD∩eCLIP reconciliation |

## Final label set (consumed by M4 training)

| Artifact | Count | Role |
|---|---:|---|
| `kd_dpsi.parquet` | 3,170,934 rows | signed per-site ΔPSI over **185 RBPs** (= 2 × 1,585,467 events; 27,172 unique regulated sites) |

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
  One row per (site, RBP). The corpus is a dense (site × regulator) → ΔPSI matrix:
  the same ~13.6K events are each probed by ~185 knockdowns.
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
- **K562/HepG2 cancer lines** — the right signal for the cancer-induced M4 arm
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

- Design + findings: [`../../meta_layer/docs/M4/`](../../meta_layer/docs/M4/)
  (`m4_conditional_design.md`).
- Sibling label workflow: [`../m3/README.md`](../m3/README.md) (novel splice sites).

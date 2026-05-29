# M3 training data & labels

What a single M3 training example is, what the labels mean, how junction
evidence relates to the labels, and the **validation status** of every pool.
Companion to [`m3_design.md`](m3_design.md) (the modeling decisions) and
[`label_audit_A1.md`](label_audit_A1.md) (the A1 audit). Data lives in
`data/mane/GRCh38/m3_labels/`; producing scripts in
[`examples/data_preparation/m3/`](../../../data_preparation/m3/).

## 1. Anatomy of an M3 training example

M3 forks the M2-S sequence model (`MetaSpliceModel`, 3-class per position). A
training example is a **genomic position** with:

- **Inputs** (extracted in Phase C, the pod job — *not yet built*): a sequence
  window + base-model (OpenSpliceAI) scores + multimodal channels
  (conservation, epigenetic, RBP `rbp_n_bound`, chromatin). **The `junction`
  modality is dropped from inputs** (see §3).
- **Label** — one of 4 values:

| Label | Meaning | Source pool |
|---|---|---|
| `donor` | novel 5′ splice site | positives (donor) |
| `acceptor` | novel 3′ splice site | positives (acceptor) |
| `neither` (0) | true non-site | negatives (hard + easy) |
| `ignore` (masked) | annotated splice site — excluded from the loss | annotation mask |

The label is **categorical** (donor / acceptor / neither), *not* a junction
read count.

## 2. The pools (current, on disk, validated 2026-05-27)

| Pool | File | Count | Role |
|---|---|---:|---|
| Positives | `positives_pooled.parquet` | **154,113** | novel sites → `donor`/`acceptor` |
| Negatives | `negatives.parquet` | **308,000** | `neither` (154K hard + 154K easy) |
| Annotation mask | `annotation_mask.parquet` | **825,746** | annotated sites → loss ignore-index |
| Disease anchors | `disease_anchors.parquet` | **6,351** | **held-out** (Phase D2); anti-joined out of training |

Positives: 93,162 acceptor + 60,951 donor; 78,006 `+` / 76,107 `−`.
Schema (key cols): `chrom, position, strand, splice_type, sources,
in_gtex_novel, in_splicevault, sv_freq_pct, dinuc`.

### Positive provenance
- **SpliceVault** (~153,855): empirically observed cryptic donor/acceptor events
  across 335K GTEx+SRA RNA-seq samples, absent from annotation. The bulk.
- **GTEx-novel** (748): GTEx junction sides surviving the cross-annotation audit
  + depth/tissue/GT-AG filters.
- (40 in both.) All annotation-clean (GENCODE ∪ RefSeq-curated removed).

### Negative composition
- **Hard (154K):** positions carrying a canonical GT/AG dinucleotide but neither
  annotated nor novel — force the model past the bare dinucleotide.
- **Easy (154K):** random non-canonical gene-body positions.

## 3. Is junction data the label? (the precise answer)

**Junction (split-read) evidence is the *source* of the positive labels, not an
input feature and not the label *value*.**

- Both positive arms are junction-evidence-derived: GTEx-novel = GTEx split-read
  junction sides; SpliceVault = empirically observed mis-splicing junctions
  across 335K samples. So junction evidence is **how we know a position is a
  real novel splice site** → it determines which positions become positive.
- The label assigned is the **categorical** `donor`/`acceptor` (or `neither` /
  `ignore`), **not** a per-position junction count or PSI regression target.
- The `junction` **modality is removed from the model inputs** — including it
  would leak the supervision target. (M1/M2 use junction as an input feature;
  M3 does not.) See [`m3_design.md`](m3_design.md) §2.

This is the sense in which the original prerequisites doc's "junctions as label,
not feature" holds: junction evidence defines the label set; junctions are not
fed to the model.

## 4. Validation status — what IS and ISN'T validated

### Validated ✓ (coordinate / dinucleotide accuracy + set integrity)
The **GT/AG-by-strand dinucleotide oracle** (donor→GT, acceptor→AG,
transcript-oriented, split by strand) is our coordinate-accuracy check. Current
results:

| Check | Result |
|---|---|
| Positives at canonical GT/AG | **1.00 both strands** (filtered to canonical) |
| Hard negatives canonical | 1.00 both strands (canonical-by-construction) |
| Easy negatives canonical | 0.00 both strands (non-canonical-by-construction) |
| SpliceVault offset reconstruction | validated by the oracle → **100% canonical** post-conversion (the snap-to-nearest-canonical step) |
| GTEx junction convention | 97.7% GT/AG both strands (no strand asymmetry) |
| Minus-strand annotation bug | **fixed** (was 0.5–0.6 on `−`; now ~0.98); positives now strand-balanced (78K`+`/76K`−`) |
| Positives ∩ annotation | 0 (genuinely novel) |
| Negatives ∩ (annotation ∪ positives) | 0 |
| Anchors ∩ training pool | 0 (no train/eval leakage) |

So **coordinate accuracy — including the SpliceVault-derived coordinates — is
validated**: every positive sits at a real splice-signal dinucleotide on the
correct strand, and the offset reconstruction was gated on that oracle.

### Functional validation — Phase B3 DONE (2026-05-28)
The dinucleotide oracle only confirms a position *looks like* a splice site. **B3**
closes the functional gap with an **independent ENCODE4 long-read** truth set: 56
tissue-diverse transcriptome GTFs → 1,309,595 canonical long-read splice sites
(120K noisy non-canonical dropped; convention verified offset-0 ≫ ±1). Built by
[`../../../data_preparation/m3/10_build_longread_truth.py`](../../../data_preparation/m3/10_build_longread_truth.py);
outputs in `data/encode_longread/GRCh38/`.

**Positive-pool confirmation (fraction appearing in long-read transcripts):**

| Arm | Confirmed | Read |
|---|---:|---|
| **GTEx-novel** | **667/707 (94.3%)** | strong — within-project novel sites are functionally real |
| GTEx-novel ∩ SpliceVault | 36/37 (97.3%) | strong |
| **SpliceVault** | 77,176/153,369 (50.3%) | **coverage-limited lower bound** — SpliceVault spans 335K short-read samples; our 56 long-read biosamples can't express them all. The confirmed half is validated; unconfirmed = "not seen in these 56 tissues," not "artifact." |
| **Disease anchors** (held-out) | **6,196/6,351 (97.6%)** | strong |
| **Pool total** | 77,879/154,113 (50.5%) | dominated by SpliceVault's coverage-limited rate |

A `longread_confirmed` (bool) + `longread_n_biosamples` (int) column is now on
`positives_pooled.parquet` — usable as a high-confidence training subset
(52,079 confirmed in ≥2 tissues) and for stratified eval.

The **anti-circular D1 eval truth set** = `longread_truth_novel.parquet`
(**681,809** long-read-confirmed sites absent from annotation): M3 will be
scored against these, not against "absent from annotation."

### Still open ⚠️
- **Negative purity** — negatives are *set-defined* non-sites (excluded from
  annotation ∪ novel ∪ long-read sites can be added), not *functionally*
  confirmed; bounded by the large pool, revisit if precision is suspect.

(Cancer-cell-line RBP coverage bias is a separate, documented consideration —
see the RBP tutorial's "What M3 actually uses" note; it's a coverage gap, not a
coordinate-accuracy issue, and the label side is not cancer-derived.)

## 5. Next steps
1. ~~**B3** — ENCODE4 long-read truth set~~ **DONE** (2026-05-28; see §4).
2. **Phase C (pod)** — extract multimodal features at the labeled positions
   (bigWig streaming) → gene/position cache → train `MetaSpliceModel` (M3,
   junction dropped). Same bigWig cost as M1/M2 → a pod job. Consider training
   on / up-weighting the `longread_confirmed` high-confidence subset.
3. **D1** — score M3 against `longread_truth_novel.parquet` (anti-circular).
4. **D2** — evaluate generalization on the held-out disease anchors.

## Related
- Data workflow + run order: [`../../../data_preparation/m3/README.md`](../../../data_preparation/m3/README.md)
- Modeling decisions: [`m3_design.md`](m3_design.md) · A1 audit: [`label_audit_A1.md`](label_audit_A1.md)
- Output index: `output/meta_layer/m3_label_audit/README.md`

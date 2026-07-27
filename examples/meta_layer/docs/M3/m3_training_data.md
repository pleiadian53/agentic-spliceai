# M3 training data & labels

What a single M3 training example is, what the labels mean, how junction
evidence relates to the labels, and the **validation status** of every pool.
Companion to [`m3_design.md`](m3_design.md) (the modeling decisions) and
[`label_audit_A1.md`](label_audit_A1.md) (the A1 audit). Data lives in
`data/mane/GRCh38/m3_labels/`; producing scripts in
[`examples/data_preparation/m3/`](../../../data_preparation/m3/).

> **§1–4** cover the M3 **recognizer** (v1: per-position 3-class). **§5** covers
> the M3-R **candidate refiner** — the current Tier 2 direction (base proposes
> candidates → classify real cryptic vs artifact from multimodal evidence).

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

## 5. M3-R — candidate-refiner training data (Tier 2, current direction)

The §1–4 pools train the **recognizer** (M3 v1): scan every position, is this a
novel splice site. The Tier 0 evaluation showed v1 is the best novel-site ranker
we have, **but its multimodal channels barely help in the genome-scan frame**
(+0.016 P@5 vs zeroing them). **M3-R reframes the task** so the multimodal
evidence can be decisive: the base model *proposes* candidates; a classifier
reranks each **real cryptic vs artifact**. Everything here is **local** (the
peak-preserving feature parquets already hold every candidate's vector) — no pod,
no bigWig streaming. Builder:
[`../../../data_preparation/m3/11_build_candidate_labels.py`](../../../data_preparation/m3/11_build_candidate_labels.py)
→ `data/mane/GRCh38/m3_labels/candidate_labels.parquet`.

### Anatomy of an M3-R example
A **base-proposed candidate** = a `(position, splice_type)` with its full 116-col
multimodal feature vector, pulled from
`data/mane/GRCh38/openspliceai_eval/analysis_sequences/analysis_sequences_chr*.parquet`
(peak-preserving: every position with base prob > 0.01 is kept). Label is
**binary**: `1` = real cryptic, `0` = artifact.

### Labeling
| Class | Definition | Count (confirmed run) |
|---|---|---:|
| real (1) | candidate ∈ `positives_pooled`, **`longread_confirmed`** subset (Tier 1 cleanup) | 13,105 |
| artifact (0) | base-proposed (`max(donor_prob, acceptor_prob) > 0.01`) but **not** a real/eval site, base-matched | 39,215 (~3×) |

- The candidate's `splice_type` comes from the positive's known type (positives)
  or `argmax(donor_prob, acceptor_prob)` (negatives) — **not** the parquet's
  annotation-derived `splice_type`, which is empty for non-annotated positions.

### The crux — base-score-matched hard negatives
Real novel sites are intrinsically **low base score** (median ~0.044 — that is
*why* they are cryptic). If negatives were drawn arbitrarily (as the recognizer's
`negatives.parquet` was — dinucleotide decoys with ~0 base score), a classifier
would just relearn the base score. Instead, negatives are **stratified to match
the positives' base-score histogram per splice type**, so the two classes are
base-score-indistinguishable (verified: donor pos/neg medians 0.044/0.043,
acceptor 0.045/0.044). The base score cannot separate them → the classifier is
**forced** to use conservation / epigenetic / chromatin / RBP.

### Eval-leak guard (what is excluded from negatives)
The negative pool is anti-joined against **`annotation_mask ∪ positives_pooled ∪
D1 (long-read truth) ∪ D2 (disease anchors)`** on `(chrom, position, strand)`, so
**no held-out eval-truth site can ever be labeled an artifact** (that would poison
Phase 2's D1/D2 recall — an eval site trained as fake). D2 anchors (novel *and*
annotated) are all excluded for this reason.

> Note on anti-circularity: positives are **not** anti-joined against D1/D2. The
> `longread_confirmed` positives are a *subset of D1* by construction, so removing
> them would delete the training set. Anti-circularity comes from the
> **chromosome split** (below), not from site-set disjointness — a test-chrom D1
> site is held out because its *gene* is in the test split, exactly as in Tier 0.

### Features
The 116-col multimodal block via `get_feature_columns(df,
exclude_modalities=["junction"])`, additionally dropping the raw base
probabilities (`donor_prob`/`acceptor_prob`/`neither_prob` — used to *propose* and
*base-match* candidates, not as features) and the `cand_*` bookkeeping columns.
Leaky/metadata columns (`splice_type`, `position`, gene ids, …) are excluded by
`EXCLUDE_COLS`. `junction` is dropped as the label-side modality (§3), leaving ~88
features: base-derived shape (43) + conservation + epigenetic + chromatin + RBP +
genomic.

### Hold-out / validation split
Leakage-safe **gene-level SpliceAI split** (`get_gene_split(preset="spliceai")`) —
identical to M1-P and the Tier 0 eval universe:
- **train**: genes on chr2,4,6,8,10–22,X,Y.
- **val**: 10% of train genes (XGBoost early stopping only).
- **test (held out)**: genes on **chr1,3,5,7,9** — never seen in training.

Splitting by **gene** (not by row) prevents within-gene leakage (a gene's
positions are correlated). Because the test chroms match Tier 0, both the Phase 1
held-out AUC and the Phase 2 precision@k eval are anti-circular and comparable to
the recognizer's numbers.

### Phase 0 diagnostic (go/no-go, done)
Logistic AUC(base, 43 feats) **0.811** → base+all-multimodal **0.879 (+0.068)**;
XGBoost **AUC 0.90**, **~57% of SHAP importance is non-base** (epigenetic biggest;
RBP weak — cell-type mismatch). The reframe works. ⚠️ This is on the *training
label distribution* (PU noise: some "artifacts" may be unlabeled reals) — **Phase 2
(base-vs-M3-R precision@k on D1/D2) is the real, anti-circular test.**

## 6. Next steps
1. ~~Recognizer: B3 truth set / Phase C train / D1+D2 eval~~ **DONE** (Tier 0;
   see [`m3_eval_D_results.md`](m3_eval_D_results.md)).
2. **M3-R Phase 1** — formalize `examples/meta_layer/14_train_candidate_refiner.py`
   (gene-split, XGBoost `binary:logistic`, SHAP + leave-one-modality ablation,
   save model → `output/meta_layer/m3r_candidate_refiner/`).
3. **M3-R Phase 2** — `examples/meta_layer/15_evaluate_candidate_classifier.py`:
   rerank the per-gene candidate pool by M3-R vs base, precision@k/recall@k on
   D1/D1_hiconf/D2 (reuses `_m3_novel_eval.py`). The honest test.

## Related
- Data workflow + run order: [`../../../data_preparation/m3/README.md`](../../../data_preparation/m3/README.md)
- Modeling decisions: [`m3_design.md`](m3_design.md) · A1 audit: [`label_audit_A1.md`](label_audit_A1.md)
- Output index: `output/meta_layer/m3_label_audit/README.md`

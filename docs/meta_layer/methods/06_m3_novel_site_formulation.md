# M3 — Novel Splice-Site Formulation

The methodology behind the novel-site model line: how the task is posed, what the labels are and are
not, the two model formulations (recognizer and candidate refiner), and — most importantly — the
**anti-circular evaluation methodology** that turned a hard-to-measure discovery task into a set of
honest, reproducible numbers. This is the M3 counterpart to
[`05_m2_variant_formulations.md`](05_m2_variant_formulations.md); the *results* it produces live in
[results/m3_novel.md](../results/m3_novel.md), and the *ops steps* to reproduce them are the
[M3 workflow sub-series](../../workflows/meta_layer/09_m3_label_curation.md).

!!! abstract "The one idea to keep"
    A novel cryptic donor and an annotated canonical donor are **sequence-identical**. "Novel" is not a
    biological property the model can learn — it is *absence from a database*. So M3 never learns
    novelty; it learns to **recognize splice sites**, and novelty is **applied at inference** by exact
    set-subtraction against the annotation. Everything below follows from that.

---

## 1. Recognizer + post-filter (novelty is applied, not learned)

The naïve framing — "train a classifier to output *novel site: yes/no*" — is incoherent. If a novel
cryptic donor and an annotated canonical donor have the same sequence and the same local context, no
function of the inputs can separate them; the only thing that differs is a lookup in an annotation set,
which is masked out of any honest training signal. Labeling annotated donors as negatives while novel
donors are positives creates **contradictory labels at identical motifs**.

M3 resolves this by splitting the problem:

1. **Recognize** — a model scores every position for splice-site potential (donor / acceptor / neither),
   exactly like M1-S/M2-S.
2. **Post-filter** — at inference, exact set-subtraction against the annotation union
   (GENCODE ∪ RefSeq-curated) turns the recognizer's output into *novel* calls. A predicted donor that
   is already annotated is simply removed.

Novelty is a property of the **output**, not the model. This is the single most important design
decision in the M3 line and it recurs in every downstream choice (loss masking, evaluation, the
candidate refiner).

---

## 2. Junction reads are the label, not a feature

M1-S and M2-S take **junction support** (GTEx split reads) as an *input* channel — it is one of their
strongest signals. M3 cannot: junction evidence is precisely *how we know a position is a real splice
site the annotation missed*, so it **is** the supervision signal. Using it as an input would leak the
target (`junction_has_support → junction_has_support`).

So in M3 the junction modality is **removed from the inputs** (`mm_channels = 7` instead of 9), and
junction evidence instead defines the **positive label set**. The remaining channels — sequence, base
scores, conservation, epigenetic, chromatin, RBP — are what the model uses to *recognize* the site.

!!! warning "This is not a `junction_has_support` regression"
    An earlier framing posed M3 as regressing a per-position `junction_has_support` target. The model
    that was actually built and evaluated is the **recognizer + post-filter** above: a 3-class splice
    recognizer whose positives are junction-derived novel sites, with novelty applied by set-subtraction.
    The label is categorical (donor / acceptor / neither), not a junction count.

---

## 3. The label pools

Everything is coordinate-keyed `(chrom, position, strand, splice_type)`, bare chrom, under
`data/mane/GRCh38/m3_labels/` (curation steps in the
[label-curation workflow](../../workflows/meta_layer/09_m3_label_curation.md)).

| Pool | File | Count | Role |
|------|------|------:|------|
| Positives (novel) | `positives_pooled.parquet` | 154,113 | recognizer positives → donor / acceptor |
| — long-read-confirmed subset | (`longread_confirmed` flag) | 77,879 | high-confidence subset; the Tier 1 label test |
| Annotation mask | `annotation_mask.parquet` | 825,746 | annotated sites → **ignore-index** in the loss; and the post-filter set |
| Negatives (recognizer) | `negatives.parquet` | 308,000 | GT/AG decoys + easy non-sites (per-gene-window training makes these largely vestigial — see §4) |
| Candidate labels (refiner) | `candidate_labels.parquet` | 52,320 | base-score-matched real-vs-artifact table for M3-R (§5) |
| D1 truth | `…/encode_longread/…/longread_truth_novel.parquet` | 681,809 | **independent** eval truth (§6) |
| D2 anchors | `disease_anchors.parquet` | 6,351 | held-out disease-relevant eval truth (§6) |

Positives are dominated by **SpliceVault** (~153.8K cryptic events observed across 335K RNA-seq samples,
GRCh38-native, 100% GT/AG) plus a small **GTEx-novel** survivor set; both are junction-derived and
annotation-clean. Disease anchors (TDP-43 / SF3B1 / ENCODE-KD) are held out of training entirely.

---

## 4. Recognizer formulation (M3-S)

M3-S forks the M2-S backbone (v3 `MetaSpliceConfig`, H=32, ~400 bp receptive field, three-stream CNN)
with two training-loop changes and one input change:

- **Input:** junction dropped (`mm_channels = 7`).
- **Loss masking:** annotated positions carry the sentinel label `255` and are **masked out of the loss**
  (`ignore_index`), so the model is neither rewarded nor penalized on them — this is what avoids the
  identical-motif contradiction of §1.
- **Confidence weighting:** long-read-confirmed positives get a per-position `sample_weight`
  (`--confirmed-weight`, default 2×).

**Per-gene-window, not per-site.** Labels are 3-class arrays over the whole gene window
(`0=donor, 1=acceptor, 2=neither, 255=ignore`), so a single forward pass supervises every position. A
per-site formulation (one forward pass per candidate) wastes ~5000× the gradient signal on this dilated
CNN and was abandoned. This is why `negatives.parquet` is largely vestigial for the recognizer: every
non-positive, non-ignored position in the window is class 2 automatically.

Trained variants: `m3_v1` (all positives, confirmed 2×) and `m3s_v1_1_confirmed` (Tier 1, confirmed-only).

---

## 5. Candidate-refinement formulation (M3-R)

M3-R is a **different formulation of the same task**, motivated by the finding that M3-v1's multimodal
channels barely help genome-wide (see §6). Instead of scanning the genome, let the base model **propose**
candidates (positions above a low score threshold, novel after post-filter), then train a classifier to
answer *real cryptic site vs artifact* from the multimodal evidence.

The load-bearing methodological choice is **base-score-matched hard negatives**. Real novel sites are
intrinsically low base score (median ~0.04 — that is *why* they are cryptic). If negatives were drawn
arbitrarily, the classifier would just relearn the base score. Instead, negatives are sampled from the
same candidate pool and **stratified to match the positives' base-score histogram per splice type**, so
the base score cannot discriminate and the multimodal channels are *forced* to earn their keep.

- Positives = confirmed novel sites present in the feature parquets.
- Negatives = base-proposed positions **not** in `annotation ∪ positives ∪ D1 ∪ D2` (the eval sets are
  excluded so no held-out truth site is ever trained as an artifact), base-matched.
- Model = XGBoost `binary:logistic`; features = the multimodal block minus junction and the raw base
  probabilities; leakage-safe **gene-level** SpliceAI split.

This is entirely local (the peak-preserving feature parquets already hold every candidate's vector).
Full training-data construction: [label-curation workflow §candidate labels](../../workflows/meta_layer/09_m3_label_curation.md).

---

## 6. Anti-circular evaluation (the core contribution)

Evaluating a novel-site model against the annotation it trained on is **circular** — "sites not in
Ensembl" is how the positives were defined. M3's evaluation methodology is the part most worth reusing:

**Independent truth.** Score against evidence the model never trained on — **D1** (ENCODE long-read
novel junctions; independent of the SpliceVault short-read signal) and **D2** (held-out disease anchors).
D1 splits into `D1_hiconf` (≥ 2 biosamples) for a stricter variant.

**Held-out chromosomes.** The eval universe is the SpliceAI test set (chr 1, 3, 5, 7, 9), held out for
M1-S / M2-S / M3-v1 alike — so the comparison is clean without further site-set surgery. (Note the
positives are a *subset* of D1 by construction, so anti-circularity comes from the **chromosome split**,
not from making the sets disjoint.)

**The metric is per-gene precision@k / recall@k — a *within-gene* ranking metric.** "Of the top-k novel
candidates *in this gene*, how many are real?" This matches how a discovery pipeline is actually run
(start from a locus, shortlist candidates) and is deliberately **not** genome-wide PR-AUC, which the
~99.99%-negative genome makes uninformative.

**Novelty post-filter is part of the metric.** Before ranking, annotated positions are set-subtracted
(masked to `−inf`), so precision@k measures novelty recovery, not "is this a canonical site."

### Within-gene vs between-gene — the decomposition that explains M3-R

M3-R trains to AUC 0.90 yet **ties base** on per-gene precision@k. Decomposing the AUC resolves it:

| AUC | Base | M3-R |
|-----|------|------|
| Global (pooled across all candidates) | 0.585 | 0.675 (+0.09) |
| Within-gene (mean of per-gene AUCs) | 0.602 | 0.592 (−0.01) |

M3-R's advantage is **entirely between-gene** — it learned "this is the *kind of locus* where cryptic
sites occur," which is genuine information but not what within-gene ranking rewards. The root cause is
mechanistic: the genome-averaged multimodal tracks are approximately **constant across a gene's
candidate positions**, so they carry locus-level, not position-level, signal. This single decomposition
unifies every "multimodal barely helps M3" observation across the line and is the methodological reason
the forward path is *position-level* features rather than more genome-averaged tracks.

---

## 7. The Tier ladder — a methodology for honest development

M3 improvement was structured as a ladder, each rung a cheap experiment that gates the next:

- **Tier 0 — measure the real thing.** Build the anti-circular eval before optimizing anything. It
  replaced M3-v1's circular in-distribution PR-AUC (0.30) with the honest per-gene numbers and revealed
  M3-v1 is actually the best novel ranker.
- **Tier 1 — fix the labels.** Retrain on confirmed-only positives. *Result: no help* — label noise was
  not the ceiling.
- **Tier 2 — reframe.** The candidate refiner (M3-R). *Result: honest negative* — multimodal is
  between-gene, not within-gene.

The value of the ladder is that each negative result is **precisely characterized**, not just "didn't
work." Both the metric (within-gene precision@k) and the decomposition (within vs between-gene) were
what made the negatives interpretable and pointed cleanly at the next lever.

---

## 8. What M3 established

- **M3-v1 (recognizer) is the best novel-site ranker** and the current within-gene deliverable.
- **Neither cleaner labels (Tier 1) nor the candidate-refinement reframe (Tier 2) improve within-gene
  ranking** — a fully-characterized, reusable negative.
- **Genome-averaged multimodal tracks are locus-level evidence.** They lift canonical (M1) and
  alternative (M2) recognition, where the question is partly "what kind of locus," but novel-site
  *discovery* needs position-level signal they do not provide.
- Repurposable byproduct: M3-R's global AUC (0.675) is a **gene/locus-triage** signal — which genes to
  search, not which base.

## Further reading

- [Model variants M1–M4](00_model_variants_m1_m4.md#m3-novel-splice-site-prediction) — where M3 sits in the family.
- [M3 novel-site results](../results/m3_novel.md) — the evaluated numbers and version story.
- [M3 workflow sub-series](../../workflows/meta_layer/09_m3_label_curation.md) — reproduce it end to end.
- Local dev write-ups: `examples/meta_layer/docs/M3/` (design, training data, eval results).

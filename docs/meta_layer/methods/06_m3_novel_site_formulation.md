# M3: Novel Splice-Site Formulation

This chapter explains how the novel-site model line (M3) is set up, why it is set up that way, and how it is evaluated. It assumes you know what a classifier, a train/test split, and precision/recall are. It does not assume you know anything about splicing databases or about how this project's earlier models work.

The chapter has three jobs. First, it explains how the task is posed and why "novelty" is not something a model can learn (sections 1 and 2). Second, it describes the labels and the two model formulations built on them, a recognizer and a candidate refiner (sections 3 to 5). Third, and most importantly, it describes the **anti-circular evaluation** that turned a hard-to-measure discovery task into a set of honest, reproducible numbers (sections 6 to 8). It is the M3 counterpart to [`05_m2_variant_formulations.md`](05_m2_variant_formulations.md). The *results* live in [results/m3_novel.md](../results/m3_novel.md), and the *ops steps* to reproduce them are the [M3 workflow sub-series](../../workflows/meta_layer/09_m3_label_curation.md).

!!! abstract "The one idea to keep"
    A novel cryptic donor and an annotated canonical donor can be sequence-identical. "Novel" is therefore not a biological property that a model can learn. It simply means "absent from a database". So M3 never learns novelty. It learns to **recognize splice sites**, and novelty is **applied afterwards** by checking each prediction against the annotation. Everything below follows from that.

---

## 1. Recognizer + post-filter (novelty is applied, not learned)

### The tempting framing

The most natural way to build a "novel splice-site model" is to train a binary classifier whose output is *novel site: yes or no*. Annotated sites would be the "no" examples, and unannotated-but-real sites would be the "yes" examples.

This framing breaks down on a concrete case. Take two donor sites in the same gene. Site A is in the annotation. Site B is not, but it is a genuine cryptic donor that RNA-seq reads confirm. Both sit in the same kind of local sequence, with a `GT` dinucleotide and similar flanking bases. Whatever signals the model can see (sequence, conservation, chromatin, RBP binding), they look alike. The only thing that differs is a database lookup: A is in the table and B is not. That lookup is not an input the model receives, and it should not be, because it is precisely what we want to discover.

If we label A as a negative and B as a positive anyway, we ask the model to separate two near-identical inputs. The best it can do is hedge between the two answers. These are **contradictory labels at identical motifs**.

### The split that resolves it

M3 divides the problem into two steps, only one of which involves learning:

1. **Recognize.** A model scores every position in a gene for splice-site potential: donor, acceptor, or neither. This is exactly the job M1-S and M2-S do, and it is a well-posed problem because a real splice site is a real splice site whether or not a database lists it.
2. **Post-filter.** At inference time, remove every predicted site that appears in the annotation union (GENCODE ∪ RefSeq-curated). What survives is the list of *novel* calls. This is an exact set subtraction, with no learning involved.

Novelty is a property of the model's **output**, not of the model. This is the most important design decision in the M3 line, and it returns in every later choice: the loss masking in section 4, the candidate refiner in section 5, and the evaluation in section 6.

> **Recap.** Learn "is this a splice site?" from data. Decide "is it novel?" by lookup. Never ask a model to learn the lookup.

---

## 2. Junction reads are the label, not a feature

### What a junction read is

When a cell splices a transcript, an RNA-seq read that happens to cover the join gets split: part of it aligns to the exon before the intron, and part to the exon after. Such a **split read** pinpoints a donor coordinate (end of the upstream exon) and an acceptor coordinate (start of the downstream exon). **Junction support** is the count of split reads at a coordinate, and here it comes from GTEx.

### How M1-S and M2-S use it

For M1-S and M2-S, junction support is an **input channel**, and it is one of their strongest signals. If many reads splice at a position, the position is probably a splice site, and that is useful evidence for deciding whether a borderline candidate is real.

### Why M3 cannot use it

M3's positives were found from junction evidence in the first place: it is how we know that a position is a real splice site the annotation missed. Junction evidence is therefore the **supervision signal**. If it were also an input, the model would receive, in one column, nearly the same measurement that defined the answer. The model could do well by reading that column and learn little about sequence or genomic context, which is the opposite of what we want from a recognizer. The input and the label would be two views of the same observation.

So M3 does two things:

- It **removes the junction modality from the inputs**. The model sees 7 multimodal channels (`mm_channels = 7`) instead of the 9 used by M1-S and M2-S. The nine are PhyloP and PhastCons conservation, two histone marks, ATAC and DNase accessibility, two junction channels, and an eCLIP RBP-binding count. Dropping the two junction channels leaves seven.
- It uses junction evidence to **define the positive label set**.

The remaining channels (sequence, base-model scores, conservation, epigenetic marks, chromatin accessibility, RBP binding) are what the model uses to *recognize* the site.

!!! warning "This is not a `junction_has_support` regression"
    An earlier framing posed M3 as regressing a per-position `junction_has_support` target. The model that was actually built and evaluated is the recognizer plus post-filter from section 1. It is a 3-class splice recognizer whose positives are junction-derived novel sites, with novelty applied by set subtraction. The label is categorical (donor / acceptor / neither), not a junction count.

---

## 3. The label pools

Before the table, two conventions. Every site is identified by the key `(chrom, position, strand, splice_type)`, and `chrom` is written bare (`1`, not `chr1`). All files live under `data/mane/GRCh38/m3_labels/` unless noted, and the curation steps are in the [label-curation workflow](../../workflows/meta_layer/09_m3_label_curation.md).

The pools have different jobs. **Positives** are what the recognizer learns to find. The **annotation mask** serves twice: it marks positions the loss ignores during training, and it is the set subtracted at inference. **Negatives** and **candidate labels** support the two formulations in sections 4 and 5. The **D1** and **D2** truth sets are never trained on, and exist only for evaluation (section 6).

| Pool | File | Count | Role |
|------|------|------:|------|
| Positives (novel) | `positives_pooled.parquet` | 154,113 | recognizer positives → donor / acceptor |
| of which long-read-confirmed | (`longread_confirmed` flag) | 77,879 | high-confidence subset; the Tier 1 label test (section 7) |
| Annotation mask | `annotation_mask.parquet` | 825,746 | annotated sites → **ignore-index** in the loss; also the post-filter set |
| Negatives (recognizer) | `negatives.parquet` | 308,000 | GT/AG decoys + easy non-sites (largely vestigial under per-gene-window training, see section 4) |
| Candidate labels (refiner) | `candidate_labels.parquet` | 52,320 | base-score-matched real-vs-artifact table for M3-R (section 5) |
| D1 truth | `…/encode_longread/…/longread_truth_novel.parquet` | 681,809 | **independent** eval truth (section 6) |
| D2 anchors | `disease_anchors.parquet` | 6,351 | held-out disease-relevant eval truth (section 6) |

The positives are dominated by **SpliceVault**: about 153.8K cryptic splicing events observed across roughly 335K RNA-seq samples, GRCh38-native, and 100% GT/AG. A small set of **GTEx-novel** survivors adds to it. Both sources are derived from junction reads and are annotation-clean. The disease anchors (TDP-43, SF3B1, ENCODE-KD) are held out of training entirely.

!!! warning "The anchor *pool* spans three mechanisms; the D2 *eval set* does not"
    The three sources above describe the 6,351-site pool. What survives the held-out chromosome filter to become D2 is **171 SF3B1 + 2 ENCODE-KD + 0 TDP-43**, and **171 of those 173 are acceptors**. In practice D2 is an SF3B1 benchmark. See [section 6](#6-anti-circular-evaluation-the-core-contribution) for what that means when reading the number.

---

## 4. Recognizer formulation (M3-S)

M3-S starts from the M2-S backbone: the v3 `MetaSpliceConfig` architecture, hidden width 32, and a three-stream convolutional network that sees about 400 bp of context around each position. It differs from M2-S in two training-loop changes and one input change.

- **Input change.** The junction channels are dropped (`mm_channels = 7`, section 2).
- **Loss masking.** Annotated positions carry the sentinel label `255` and are **masked out of the loss** (PyTorch's `ignore_index`). The model is neither rewarded nor penalized on them. This is what avoids the identical-motif contradiction from section 1: annotated sites are simply not part of the learning signal.
- **Confidence weighting.** Long-read-confirmed positives get a larger per-position `sample_weight` (`--confirmed-weight`, default 2×), so that better-supported labels count for more.

### Per-gene-window, not per-site

Labels are arrays over a whole gene window with four values: `0` = donor, `1` = acceptor, `2` = neither, `255` = ignore. For a toy 8-position window containing one annotated donor and one novel acceptor, the label array looks like this:

```text
position:   0    1    2    3    4    5    6    7
label:      2    2   255   2    2    1    2    2
                      ^              ^
                annotated site    novel acceptor
```

One forward pass over the window then supervises every position at once. A per-site formulation (one forward pass per candidate) wastes roughly 5000× the gradient signal on this dilated CNN, so it was abandoned. This also explains why `negatives.parquet` is largely vestigial for the recognizer: every position in the window that is neither a positive nor ignored is automatically class 2.

Trained variants:

- `m3_v1`: all positives, with confirmed sites weighted 2×.
- `m3s_v1_1_confirmed`: the Tier 1 retrain on confirmed-only positives (section 7).

---

## 5. Candidate-refinement formulation (M3-R)

### The motivation

M3-R is a **different formulation of the same task**. It exists because of a finding in section 6: the multimodal channels barely help M3-v1 when it scans a whole gene. M3-R tries another route. Rather than scanning every position, let the base model **propose** candidates (positions scoring above a low threshold that survive the novelty post-filter), then train a classifier to answer one narrower question: *is this candidate a real cryptic site or an artifact?* The hope is that the multimodal evidence can decide that question even when it could not rank positions.

### The load-bearing choice: base-score-matched negatives

Real novel sites have intrinsically low base scores (median about 0.04). That is *why* they are cryptic: the base model barely notices them. Suppose the negatives were drawn arbitrarily from the candidate pool. Most would have even lower scores, and a classifier could separate the two groups by reading the base score alone. It would look accurate and teach us nothing about the multimodal channels.

So negatives are sampled from the same candidate pool and **stratified to match the positives' base-score histogram, separately for donors and acceptors**. If 30% of the positives fall in a given score band, 30% of the negatives are drawn from that band. The base score then cannot discriminate between the classes, and the multimodal channels are *forced* to earn their keep.

### The training recipe

- **Positives:** confirmed novel sites that are present in the feature parquets.
- **Negatives:** base-proposed positions that are **not** in `annotation ∪ positives ∪ D1 ∪ D2`, base-matched as above. Excluding D1 and D2 guarantees that no held-out truth site is ever trained on as an "artifact".
- **Model:** XGBoost with a `binary:logistic` objective.
- **Features:** the multimodal block minus the junction channels and the raw base probabilities.
- **Split:** gene-level, following the SpliceAI split convention, so that nearby positions of one gene never appear on both sides of the split.

The whole procedure is local, because the peak-preserving feature parquets already hold every candidate's feature vector. The full construction of the training data is in the [label-curation workflow](../../workflows/meta_layer/09_m3_label_curation.md).

---

## 6. Anti-circular evaluation (the core contribution)

### What circular means here

"Novel" was defined as "not in Ensembl", and the positives were selected using that definition. Scoring a novel-site model against the same annotation it trained against would therefore be **circular**: the model is graded on the exercise it was built from. An earlier in-distribution PR-AUC of 0.30 is an example. It showed that training ran, but not that M3 finds sites that independent evidence supports. M3's evaluation is built to avoid this, and the method is the part of this chapter most worth reusing.

### Independent truth

The model is scored against evidence it never trained on.

- **D1** is ENCODE long-read novel junctions. Long reads observe whole transcripts directly, so D1 is independent of the short-read SpliceVault signal that produced the positives. D1 also has a stricter variant, **D1_hiconf**, that requires support from at least 2 biosamples.
- **D2** is the set of held-out disease anchors.

### What D2 actually asks

D2 is 171 SF3B1 anchors, 2 ENCODE-KD, and no TDP-43 (section 3). That matters beyond provenance, because the mechanism sets the *shape* of the task.

SF3B1 is a spliceosome component. Its mutations shift which branch point is used, so the cryptic acceptor sits only a few nucleotides from the canonical one. Measured against the annotation, every SF3B1 anchor is within **50 nt** of an annotated site of the same type, with a median of **17 nt**. A TDP-43 cryptic exon is a different kind of object. It is a repressed cassette exon sitting **hundreds to tens of thousands of nt** inside an intron. The three curated ALS sites sit at 419, 431 and 19,980 nt.

So D2 measures "*can the model rank a near-competitor of a known acceptor?*". That is a real and useful question, but it is **not** "*can it find a cryptic exon in the middle of an intron?*". Quote the D2 number as evidence for the first. The second is unmeasured, because the curated TDP-43 sites lie on training chromosomes and so cannot enter a held-out evaluation at all.

### Held-out chromosomes

The evaluation universe is the SpliceAI test set (chromosomes 1, 3, 5, 7, 9), which is held out for M1-S, M2-S and M3-v1 alike. The comparison between models is therefore clean without any further surgery on the site sets. One subtlety: the positives are a *subset* of D1 by construction, so the anti-circularity comes from the **chromosome split**, not from the two sets being disjoint.

### The metric: per-gene precision@k and recall@k

The metric is shaped like the use case. A discovery pipeline starts from a locus and asks for a shortlist of candidates to inspect. So for each gene, rank the novel candidates by model score and look at the top $k$:

- **precision@k** is the fraction of the top $k$ that are real: $\text{hits in top } k \,/\, k$.
- **recall@k** is the fraction of that gene's real sites found in the top $k$: $\text{hits in top } k \,/\, \text{number of truth sites in the gene}$.

These are averaged over genes. They are deliberately **not** genome-wide PR-AUC, which the ~99.99%-negative genome makes uninformative. It is a **within-gene ranking** metric.

**Worked example.** This is the fixture the evaluation code uses to test itself. A donor-scoring gene has four high-scoring positions:

| Position | Model score | Status |
|---------:|------------:|--------|
| 105 | 0.90 | annotated (will be filtered out) |
| 110 | 0.80 | truth, novel |
| 118 | 0.70 | not in truth, not annotated (false positive) |
| 122 | 0.60 | truth, novel |

The post-filter removes position 105 first. The ranked novel candidates are then 110, 118, 122, and there are 2 truth sites in the gene. So:

| $k$ | Top-$k$ | Hits | precision@k | recall@k |
|----:|---------|-----:|------------:|---------:|
| 1 | 110 | 1 | 1.000 | 0.5 |
| 2 | 110, 118 | 1 | 0.500 | 0.5 |
| 3 | 110, 118, 122 | 2 | 0.667 | 1.0 |

### The novelty post-filter is part of the metric

Before ranking, annotated positions are set-subtracted (their score is set to $-\infty$). Without this step, every model would rank the canonical annotated sites at the top and precision@k would be meaningless. With it, precision@k measures **novelty recovery**, not "is this a canonical site".

### Within-gene vs between-gene: the decomposition that explains M3-R

M3-R trains to an AUC of 0.90, yet it only **ties the base model** on per-gene precision@k. Both facts are true, and a decomposition of the AUC shows how. AUC is the probability that a randomly chosen real site is scored above a randomly chosen artifact, so 0.5 is chance and 1.0 is perfect. It can be computed two ways:

- **Global AUC** pools all candidates from all genes into one ranking.
- **Within-gene AUC** computes an AUC inside each gene and averages them.

| AUC | Base | M3-R |
|-----|------|------|
| Global (pooled across all candidates) | 0.585 | 0.675 (+0.09) |
| Within-gene (mean of per-gene AUCs) | 0.602 | 0.592 (−0.01) |

M3-R's improvement is **entirely between-gene**. It learned "this is the *kind of locus* where cryptic sites occur", which is genuine information. But per-gene precision@k only rewards telling candidates apart *inside* one gene, and there M3-R is no better than the base.

The root cause is mechanistic. The multimodal tracks (conservation, histone marks, accessibility, RBP binding) are genome-wide averages, so they are approximately **constant across the candidate positions of a single gene**. An analogy: a ZIP code tells you a lot about which neighborhood a house is in, and nothing about which house on the street to pick. The tracks carry locus-level signal, not position-level signal. This one observation explains every "multimodal barely helps M3" result in the line, and it is the methodological reason the forward path is *position-level* features rather than more genome-averaged tracks.

---

## 7. The Tier ladder: a methodology for honest development

M3 improvement was organized as a ladder. Each rung is a cheap experiment, and its result decides whether the next rung is worth climbing.

- **Tier 0: measure the real thing.** Build the anti-circular evaluation before optimizing anything. It replaced M3-v1's circular in-distribution PR-AUC (0.30) with the honest per-gene numbers, and it revealed that M3-v1 is actually the best novel ranker available.
- **Tier 1: fix the labels.** Roughly half the positives are not long-read-confirmed, so label noise was the natural suspect. Retraining on confirmed-only positives *did not help*, so label noise was not the ceiling.
- **Tier 2: reframe the task.** The candidate refiner (M3-R, section 5) tried a different question. *Result: an honest negative.* Its multimodal gain is between-gene, not within-gene (section 6).

The value of the ladder is that each negative result is **precisely characterized**, not just "didn't work". The metric (within-gene precision@k) and the decomposition (within vs between-gene) are what made the negatives interpretable and pointed at the next lever.

---

## 8. What M3 established

- **M3-v1 (the recognizer) is the best novel-site ranker** and the current within-gene deliverable.
- **Neither cleaner labels (Tier 1) nor the candidate-refinement reframe (Tier 2) improves within-gene ranking.** This is a fully characterized, reusable negative.
- **Genome-averaged multimodal tracks are locus-level evidence.** They lift canonical (M1) and alternative (M2) recognition, where part of the question is "what kind of locus is this?". Novel-site *discovery* needs position-level signal that they do not provide.
- **A repurposable byproduct:** M3-R's global AUC (0.675) is a **gene/locus-triage** signal. It says which genes to search, not which base.

## Further reading

- [Model variants M1–M4](00_model_variants_m1_m4.md#m3-novel-splice-site-prediction): where M3 sits in the family.
- [M3 novel-site results](../results/m3_novel.md): the evaluated numbers and the version story.
- [M3 workflow sub-series](../../workflows/meta_layer/09_m3_label_curation.md): reproduce it end to end.
- Local dev write-ups: `examples/meta_layer/docs/M3/` (design, training data, eval results).

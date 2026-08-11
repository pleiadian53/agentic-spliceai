# Reading the numbers

The Lab makes it easy to produce a number and easy to misread one. This page is the short list of
what to check first.

## Which yardstick? {#which-yardstick}

**Every score is relative to a truth set, and the truth set can decide the winner.** The genome view
now labels this on each block; read the label before quoting the number.

| Where | Scored against | Consequence |
|---|---|---|
| Classification TP/FP/FN | **the overlaid meta model's training annotation** (Ensembl for M2-S), else the base model's — shown as a `vs …` tag beside the counts, overridable with the **Ground truth** selector | correct by default |
| F1-optimal, "this gene" | the base model's annotation, canonical only | **penalises M2-S for working** — labelled in the panel |
| F1-optimal, "held-out" | the meta model's eval: Ensembl, all annotated sites | canonical sites dominate the average |
| **Alt-sites badge** | **Ensembl ∖ MANE** | the headline for M2-S |

The classification row auto-resolves because the same predictions score wildly differently by
yardstick. TARDBP, M2-S at threshold 0.9, identical predictions throughout:

| truth | base TP/FP/FN | meta TP/FP/FN |
|---|---|---|
| MANE | 9/0/1 | 10/**19**/0 |
| **Ensembl** *(auto for M2-S)* | 9/0/23 | **29/0/3** |

Those 19 "false positives" are real annotated splice sites. Scored on the annotation M2-S was
trained on, it has **none**.

!!! note "Why the delta set is recall, not TP/FP/FN"
    You cannot compute a false positive against a *subset* of truth: a call away from the subset may
    be a perfectly correct canonical call. Scoring the base model on the delta reports 9 "false
    positives" on TARDBP that are all correct MANE calls. The delta is therefore reported as
    **recall** (the alt-sites badge), and `Ensembl (all)` — which is the delta *plus* MANE — carries
    the well-defined TP/FP/FN.

The clearest illustration: on TARDBP the per-gene sweep reports base F1 **1.000** and meta **0.833**.
That is measured on MANE, where every alternative site M2-S correctly finds is scored as an error.
A metric that improves when the model stops doing its job is the wrong metric — which is why the
published M2-S result uses the **delta set** (Ensembl ∖ MANE), matching
`09_evaluate_alternative_sites.py`.

Two consequences worth internalising:

- A model can only be judged against the annotation its task is defined by.
- "F1-optimal" without a named truth set is not a claim. It is half a claim.

## 0.5 is not an operating point

Splice sites are under 0.5% of positions, so a 0.5 cutoff is arbitrary. Base and meta score
distributions differ enough that one shared threshold always flatters one of them:

| threshold | favours |
|---|---|
| 0.5 | **base** — meta's calls look like false positives |
| 0.99 | **meta** — base has started dropping true sites |

Neither is "correct". Use the two sliders, or report PR-AUC, which is threshold-free. See
[Genome View](02_genome_view.md#the-two-thresholds).

## Recall on a subset is not TP/FP/FN

The alt-sites badge reads `base 0/22 → meta 20/22`, deliberately **not** TP/FP/FN. "False positive"
is not well defined against a *subset* of truth: a call away from an alternative site may be a
perfectly correct canonical call. Recall on the delta set is the well-defined quantity.

## Per-gene numbers are exploratory

Everything on the genome view is one gene. The F1-optimal sweep is a **post-hoc optimum on the data
being displayed** — a navigation aid for the slider, not a held-out operating point. The citable
numbers are the [results pages](../meta_layer/results/README.md), over thousands of held-out genes.

## Not every annotated site is scoreable

Prediction runs over the gene span in the *model's* annotation. Where another annotation's gene is
longer, its extra sites fall outside the window and cannot be detected by anything. TARDBP: 8 of 30.
Those are excluded from denominators, with the count shown. A denominator that includes undetectable
sites understates every model.

## Dead channels {#dead-channels}

Feature extraction pulls conservation from bigWig files. A network failure used to zero-fill the
channel and write the cache anyway, leaving a gene whose features were silently wrong — and the
all-zero guard could not catch it, because it pools evidence across genes and one bad gene among
healthy ones reads as healthy.

A failed channel query now raises and the gene is **skipped rather than cached**. If a gene is
missing after a build, check the run log for `SKIPPED on channel-extraction failure` before assuming
the data is fine. Caching the bigWigs locally removes the failure entirely — see
[Getting started](00_getting_started.md#what-needs-prebuilt-data).

## Quick reference

| Symptom | Likely cause |
|---|---|
| Meta shows many FPs | threshold too low, and/or scored against MANE — check the alt-sites badge |
| Base looks perfect | threshold suits base; raise it and watch base lose true sites |
| Alt-site denominator < track count | sites outside the scored gene span |
| An annotation row is empty | GENCODE ∖ Ensembl is ~always 0 for coding genes |
| Gene 404s on the overlay | no feature cache — build it with `02_build_showcase_feature_cache.py` |
| Gene 404s on `/novel` | not in the held-out universe (chromosomes 1/3/5/7/9) |

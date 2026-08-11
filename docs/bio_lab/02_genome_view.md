# Genome View (`/genome/{gene}`)

The main workspace: per-position splice-site probabilities for one gene, optionally with a meta model
overlaid on its base model.

## The four track bands

| Band | Shows |
|---|---|
| Donor Prob | per-position P(donor), base and meta |
| Acceptor Prob | per-position P(acceptor) |
| Classification | TP / FP / FN markers at the current threshold; meta as open diamonds |
| **Annotations** | stacked ground-truth rows — see below |

Genes over 10,000 positions are downsampled for rendering, but **peak-preserving**: every position
above a small probability floor is kept, then evenly-spaced background fills the budget. Naive
every-Nth slicing would drop the sharp 1–2 position spikes that *are* the signal.

## Annotation tracks

Four rows, toggled by the checkbox panel **below the chart** (or by clicking a chart legend entry):

| Row | Default | Meaning |
|---|---|---|
| MANE | on | Canonical reference. What the base model was trained on. |
| **Ens∖MANE** | on | **The delta set** — alternative sites MANE does not contain. What M2-S exists to find. |
| Ensembl | on | All Ensembl sites. Visually contains MANE ∪ delta. |
| Gc∖Ens | off | GENCODE's additions. Essentially always 0 for protein-coding genes, so shown greyed. |

A row label reads `Ens∖MANE (10)` where the count is **sites inside the scored window**.

!!! warning "Sites outside the gene span cannot be scored"
    Prediction runs over the gene's span in the *model's* annotation. Ensembl genes are often longer:
    TARDBP is 18,184 bp in Ensembl against 12,838 in MANE, so **8 of its 30** alternative sites fall
    outside the window entirely. They are excluded from the denominator and clipped from the tracks,
    with the count stated under the badges. Counting them as misses would understate every model
    equally and for no reason.

## The two thresholds

A splice model's raw output is a probability per position; a threshold turns that into a call. The
base and meta models have **very different score distributions**, so a single shared cutoff always
scores one of them at the wrong operating point.

- **LINK checked** (default) — one cutoff for both. The first slider is labelled `Threshold`.
- **LINK unchecked** — two independent cutoffs. The first slider relabels to `Base threshold`.

**Find F1-optimal** sweeps this gene and reports both models' optima, each with its own *Use* button.
Applying a meta threshold that differs from base unchecks LINK for you.

The panel reports two blocks, and **names the truth set for each** — read that label before quoting
either:

| Block | Scored against |
|---|---|
| This gene (post-hoc) | the base model's annotation, canonical transcript only (MANE for openspliceai) |
| Held-out evaluation | the meta model's own eval — Ensembl, **all annotated sites**, 14,724 test genes |

Neither is the delta set. See [Reading the numbers](05_reading_the_numbers.md#which-yardstick).

## Reading base vs meta — worked example

Open `/genome/TARDBP`, base `openspliceai`, meta **M2-S**, threshold **0.5**.

!!! note "Ground truth auto-resolves"
    With a meta overlay active, TP/FP/FN is scored against **that model's training annotation** —
    Ensembl for M2-S — shown as a `vs ensembl` tag beside the counts. The **Ground truth** selector
    overrides it. The walkthrough below uses `truth=mane` to show *why* the default changed; set the
    selector to MANE to reproduce it.

**Scored on MANE, the meta model looks bad:**

| | TP | FP | FN |
|---|---|---|---|
| Base | 10 | 0 | 0 |
| Meta | 10 | **39** | 0 |

Same true positives, 39 extra false positives. On this reading the meta layer hurts.

**Now look at the Alt-sites badge: `base 0/22 → meta 20/22`.** Of the 39 "false positives",
**20 sit exactly on splice sites Ensembl annotates**. They are false only because the ground truth
here is MANE, which lists one transcript per gene.

**Raise the threshold to 0.9:**

| thr | base TP/FP/FN | meta TP/FP/FN | meta FPs that are annotated alt sites | alt recall base → meta |
|---|---|---|---|---|
| 0.5 | 10/0/0 | 10/39/0 | 20 / 39 | 0/22 → 20/22 |
| **0.9** | **9/0/1** | **10/19/0** | **19 / 19 — all of them** | 0/22 → **19/22** |
| 0.99 | 7/0/3 | 10/13/0 | 13 / 13 | 0/22 → 13/22 |

At 0.9 **every** meta false positive is a real annotated splice site. The unexplained ones at 0.5
were an operating-point artifact, not bad predictions. Meanwhile the base model has begun losing
*canonical* sites — 9 of 10, then 7.

**The summary:** base finds 10 sites and misses 22; meta finds the same 10, plus 19 of the 22, with
no unexplained false positives.

**Now switch Ground truth to `Ensembl (all)`** — the default when M2-S is overlaid, and the
annotation it was actually trained on. Same predictions, threshold 0.9:

| | TP | FP | FN |
|---|---|---|---|
| Base | 9 | 0 | 23 |
| **Meta** | **29** | **0** | **3** |

29 of 32 sites with **zero** false positives, against base's 9 of 32. The 19 "false positives" from
the MANE view were never errors — they were the yardstick refusing to count the model's purpose.

The base model is not broken. It was trained on MANE and is doing exactly what it was trained to do.
The meta layer adds the context that lets it see past the canonical reference.

## Gene navigation

The **Go to gene** box jumps to another gene, keeping the current base model. Genes without a
prebuilt feature cache return a 404 naming the command to build one — see
[Getting started](00_getting_started.md#what-needs-prebuilt-data).

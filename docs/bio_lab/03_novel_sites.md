# Novel Site Explorer (`/novel/{gene}`)

A different question from the [Genome View](02_genome_view.md). That page asks *"did the model get
this gene's known sites right?"* This one asks the discovery question:

> **For this gene, what are the top candidate sites that appear in no annotation at all?**

That is a ranked list, and per-gene precision@k is literally the metric M3 is evaluated on — so the
page has the same shape as the evaluation.

## What a row shows

| Column | Meaning |
|---|---|
| Rank / M3 score | the meta model's ranking of this candidate |
| Base score, base rank | what the base model gave the same position |
| GT/AG | the splice dinucleotide, strand-aware |
| Evidence badges | ENCODE long-read support, held-out disease anchors |

**The GT/AG column is a live correctness check, not decoration.** Every row landing on the canonical
dinucleotide for its strand and site type means the coordinate handling is right. It caught a real
bug during development: offsets reasoned from first principles scored 0.00, and the convention had to
be recovered empirically.

## Evidence is deliberately independent

Badges come from **ENCODE long-read transcripts** (D1) and **held-out disease anchors** (D2). Neither
is what M3 trained on.

!!! warning "SpliceVault is deliberately not shown"
    SpliceVault (`positives_pooled`) is M3's **training pool**. Displaying it as supporting evidence
    would be circular — it would say "the model found what it was trained on."

## The servable universe

Serving is restricted to the **4,956 genes on held-out chromosomes 1/3/5/7/9** — the ones M3 never
trained on. **165** of them carry a novel disease anchor.

This is a feature. You cannot accidentally demo a training gene. Familiar genes like BRCA1, TP53 and
UNC13A are on training chromosomes and are **not** available here; the page loads and shows an inline
message naming the universe.

## A good first gene: DHX29

| M3 rank | Type | M3 prob | Base prob | Base rank | GT/AG | Long-read |
|---|---|---|---|---|---|---|
| **1** | acceptor | 1.0000 | 0.0000 | **284** | AG | ✓ 8 biosamples |
| 2 | donor | 1.0000 | 0.0008 | 13 | GT | — |
| 3 | donor | 0.9999 | 0.0395 | 2 | GT | ✓ 1 |

The base model buries a real, long-read-confirmed, SF3B1-mutant cryptic site at **rank 284** with
probability ≈ 0. M3 ranks it **first**.

`ATP6V1A` is a good second (anchor at M3 rank 5 vs base rank 132). `TPR` works but is a weaker
illustration — base already ranks its anchors 3rd and 4th.

## How to read a low base score

A near-zero base score is **the point, not a failure**. These are cryptic sites: low base score is
what makes them cryptic. The gain here is re-ranking, not new evidence.

Equally, a top-ranked candidate is a **candidate**. Across the held-out set M3 recovers about **79%
of disease cryptics in the top 20 per gene** versus 51% for base, so expect roughly 1–2 of any top-5
to be long-read confirmed. It is a triage tool, not a truth oracle.

## Status

M3 is registered `status: research`, not promoted. It is deliberately kept out of the genome-view
overlay dropdown and the metrics dashboard: it is scored by per-gene precision@k on independent
truth, not by the PR-AUC those surfaces expect, and its 7-channel input would fail against the
9-channel cache they use.

Full results: [M3 — Novel Splice Sites](../meta_layer/results/m3_novel.md).

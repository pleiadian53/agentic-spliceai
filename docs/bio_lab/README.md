# Bioinformatics Lab UI — User Guide

The **Bio Lab** is the interactive front end for Agentic-SpliceAI: browse genes, run splice-site
prediction on demand, compare a meta model against its base model, and inspect novel-site candidates.
FastAPI + Jinja2 + Plotly on **port 8005**, backed by the same `src/` library as everything else — so
what you see here is what the models actually produce, not a re-implementation.

```bash
conda run -n agentic-spliceai python -m server.bio.app
# → http://localhost:8005/
```

## The four pages

| Page | Question it answers |
|---|---|
| [Gene Browser](01_gene_browser.md) (`/`) | *Which gene do I want, and what annotation am I looking at?* |
| [Genome View](02_genome_view.md) (`/genome/{gene}`) | *Did the model get this gene's **known** sites right?* |
| [Novel Site Explorer](03_novel_sites.md) (`/novel/{gene}`) | *What's here that **no** annotation knows about?* |
| [Metrics Dashboard](04_metrics.md) (`/metrics`) | *How do the models compare across all held-out genes?* |

Read [Reading the numbers](05_reading_the_numbers.md) before quoting anything from these pages. It
covers the two mistakes that are easy to make here: treating 0.5 as an operating point, and reading a
score without knowing which annotation it was measured against.

## Start here

1. **[Getting started](00_getting_started.md)** — launch, warm the caches, know what needs prebuilt data.
2. **[Gene Browser](01_gene_browser.md)** — find a gene (including by protein name, e.g. `TDP-43`).
3. **[Genome View](02_genome_view.md)** — the main workspace, and the base-vs-meta comparison.

## What is *not* here

- **Training and evaluation** live in the [Meta-Layer MLOps workflow](../workflows/meta_layer/README.md).
  The Lab consumes trained checkpoints; it never trains.
- **The driver scripts** that build the caches this UI serves are in
  [`examples/UI_integration/`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/UI_integration/).
  Those are development scripts, not user documentation.
- **Published results** are in [meta-layer results](../meta_layer/results/README.md). Numbers shown in
  the UI are per-gene and exploratory; the results pages are the held-out, citable ones.

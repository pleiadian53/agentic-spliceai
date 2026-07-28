# Stage 6 — Evaluation

**Pipeline position:** `training → ` **this stage** ` → reporting`

Evaluation measures the one thing the meta layer exists to do: **improve on the base model.** Every
metric here is reported as *meta vs base* on held-out chromosomes, so the output answers "did refining
help, and where?" rather than a standalone accuracy number.

!!! abstract "Inputs → Outputs"
    **Reads:** a trained checkpoint (`best.pt` + `config.pt`) and a `.npz` test cache.
    **Writes:** `eval_results.json` (M1-S / M2-S yardstick), `m2a_eval_results.json` (M2-S alt sites).

---

## M1-S (and the M2-S yardstick)

[`examples/meta_layer/08_evaluate_sequence_model.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/08_evaluate_sequence_model.py)
runs the streaming held-out evaluation on the SpliceAI test split (chr1, 3, 5, 7, 9 by default):

```bash
# Build the test cache once, then it is reused on subsequent runs
python examples/meta_layer/08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s_v4_cleanannot/best.pt \
    --build-cache \
    --bigwig-cache <bigwig-cache-dir>
# → output/meta_layer/m1s_v4_cleanannot/eval_results.json
```

`config.pt` is read automatically from the checkpoint's directory. The evaluator streams one gene at a
time (`StreamingEvaluator` in
[`src/…/splice_engine/eval/streaming_metrics.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/src/agentic_spliceai/splice_engine/eval/streaming_metrics.py)),
so memory stays flat across the genome.

Run the **same command on the M2-S checkpoint** to get its canonical yardstick — this is the standard
M1-vs-M2 comparison on the shared MANE test set.

### What `eval_results.json` contains

For both the meta model and the base model, aggregated over the test gene set:

- **Per-class and macro PR-AUC** (donor / acceptor / neither) — the primary metric for this imbalanced
  problem.
- **Per-class precision / recall / F1.**
- **`fn_count` / `fp_count`** and **`fn_reduction_pct` / `fp_reduction_pct`** — how many errors the meta
  model removed relative to base.
- **Top-k accuracy** at k = 0.5, 1, 2, 4 × (number of true sites) — the SpliceAI-style ranking metric.

!!! tip "Read the metrics honestly"
    Splice sites are extremely imbalanced, so judge models by **PR-AUC and top-k at matched recall**,
    and report precision/recall at an **F1-optimal** or matched-recall threshold — never raw FP/FN
    counts at a fixed 0.5 cutoff. The `--sweep-thresholds` and `--calibrate-temperature` options
    (below) exist precisely so you don't have to lean on 0.5.

---

## M2-S alternative sites (the metric that matters for M2)

The aggregate yardstick understates M2-S because most test positions are canonical. The point of M2-S
is the **alternative sites** (Ensembl `\` MANE), evaluated by
[`examples/meta_layer/09_evaluate_alternative_sites.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/09_evaluate_alternative_sites.py):

```bash
python examples/meta_layer/09_evaluate_alternative_sites.py \
    --checkpoint output/meta_layer/m2s_v4_cleanannot/best.pt \
    --annotation-source ensembl \
    --base-scores-dir data/ensembl/GRCh38/openspliceai_eval/precomputed \
    --build-cache \
    --cache-dir output/meta_layer/gene_cache_ensembl_cleanannot
# → m2a_eval_results.json  (overall + alternative_sites sub-dicts)
```

It builds the MANE reference set, subtracts it from the Ensembl sites to isolate the alternative sites,
and reports metrics **both overall and on the alternative subset**. `--annotation-source gencode`
(with a `--gtf`) gives the `m2b` GENCODE `\` MANE variant instead.

---

## Slicing the results: genome-wide, per-chromosome, per-gene

By default `08`/`09` report a **single genome-wide aggregate** over the test gene set. To slice:

| Granularity | How |
|-------------|-----|
| Genome-wide | default (SpliceAI test chroms) |
| Per-chromosome | re-run with `--test-chroms chr1` (one chrom at a time) |
| Per-gene inference | `--genes BRCA1 TP53 CFTR` |

Note that true **per-gene precision@k / recall@k** (a ranking metric per gene) is the design of the M3
novel-site evaluator (`13_evaluate_m3_novel.py`), not of `08`/`09`. For M1-S/M2-S, per-gene means
running inference on named genes, not a per-gene metric breakdown.

---

## Diagnostics (optional)

| Goal | Command / flag |
|------|----------------|
| Sweep the decision threshold | `08 … --sweep-thresholds` |
| Temperature-calibrate probabilities | `08 … --calibrate-temperature --val-cache-dir <dir>` → `eval_results_calibrated.json` |
| Channel ablation (zero out a modality) | `08 … --zero-channels <group>` → `eval_ablation_<group>.json` |
| Score arbitrary FASTA | `08 … --fasta <file> --output-format parquet` → `fasta_predictions.parquet` |
| Base-score / calibration studies | `02_calibration_analysis.py`, `03_modality_ablation.py` (tabular `-P` line) |

!!! note "Calibration is a claim that needs evidence"
    Only describe a model as "calibrated" when you have the reliability curve / ECE from the
    calibration path to back it up. Temperature scaling produces `eval_results_calibrated.json` and a
    fitted temperature — cite those, don't assume calibration from a high PR-AUC.

---

**Reference:** the exhaustive evaluation walkthrough (all modes, cache building, FASTA inference,
threshold and calibration flags) is
[`examples/meta_layer/docs/evaluation_tutorial.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/evaluation_tutorial.md),
with the four-mode framing in
[`evaluation_hierarchy.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/evaluation_hierarchy.md).

---

→ **Next: [Stage 7 — Reporting](07_reporting.md)**

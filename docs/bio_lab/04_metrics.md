# Metrics Dashboard (`/metrics`)

Aggregate model comparison across held-out genes — the counterpart to the per-gene pages.

## Two sources

- **Base-model runs** are discovered from evaluation outputs under `examples/base_layer/output/`.
- **Meta-model runs** are **registry-driven**: the dashboard lists what `settings.yaml` declares
  under `meta_models`, then loads that model's evaluation JSON. It never guesses from directory
  names, so a run appears here only if it is a registered model.

Models registered `status: research` (currently M3-S) are excluded. They are evaluated by metrics
this dashboard does not display, so listing them would show empty columns.

## Reading a comparison

Each meta model is shown against **its own** base model — the one it refines, from the registry's
`base_model` field. Comparing M2-S against a base it never saw would be meaningless.

The provenance fields tell you what a model was built from:

| Field | Example | Why it matters |
|---|---|---|
| `variant` | `M2-S` | which task: canonical / alternative / novel |
| `arch` | `concat_fusion` | the network, named for its mechanism |
| `corpus` | `cleanannot` | the data generation |
| `train_annotation` | `ensembl` | **which annotation it learned from** |
| `eval_protocol` | `alt_sites` | which truth set its headline number uses |

`train_annotation` is the one people skip and shouldn't. M1-S learned MANE; M2-S learned Ensembl.
A number is only interpretable once you know which.

## Which numbers are safe to quote

!!! danger "Never quote FP or FN at threshold 0.5"
    Splice sites are under 0.5% of positions. At a fixed 0.5 cutoff the M2-S genome-wide evaluation
    shows FN −89.8% *and* FP 1,006,348 against base 13,163. Both are real; both are artifacts of the
    operating point. Report **PR-AUC** (threshold-free) or counts **at each model's F1-optimal
    threshold**, and say which you used.

!!! warning "Precision was corrected in August 2026"
    The evaluator subsamples "neither" positions to 1% to keep PR-AUC tractable over 600M positions.
    Threshold sweeps had been counting false positives over the *retained* rows, inflating precision
    on the FP side by roughly 100×. This dragged every stored F1 optimum far too low — M1-S meta read
    0.35 where the corrected value is 0.99.

    Recall, TP and FN were never affected, because positives are never subsampled. That is exactly
    why it stayed hidden: every recall-side headline was correct. Figures on the results pages are
    corrected; older screenshots and any `FP 346` figure are not.

## Related

- [Reading the numbers](05_reading_the_numbers.md) — the general version of the cautions above.
- [Published results](../meta_layer/results/README.md) — the citable held-out figures.
- [Reporting & promotion](../workflows/meta_layer/07_reporting.md) — how a run becomes a registered model.

# UI Integration — driver scripts (use cases)

Driver scripts (practical use cases) for integrating the **meta-layer models**
(M1-S, M2-S, … eventually **M1–M4 and beyond**) into the **Bioinformatics Lab
UI** (`server/bio/`, port 8005).

These are not toy/POC scripts. Following the project convention, `examples/`
and `notebooks/` are where features are developed as concrete, runnable use
cases; once matured they are grouped and refactored into applications (and,
when mature, products). This directory groups **all UI design-related example
scripts** in one place — both ones that exercise the *existing* base-model UI
and ones that build toward the meta-model overlay.

**Assumption:** meta models are already pre-trained and their weights are
available under `output/meta_layer/` (current production = the leakage-clean
`m1s_v3_neuronal/` and `m2s_v3_neuronal/`, trained with full RBP). These
scripts do not train models; they consume trained checkpoints.

## Why this lives outside `examples/meta_layer/`

The meta-layer training/evaluation scripts (`examples/meta_layer/`) are about
*producing* models. UI integration is a different concern — *serving* trained
models behind the Lab UI — so it gets its own topic directory.

## Scripts

| # | Script | Purpose | Status |
|---|--------|---------|--------|
| 01 | `01_bio_ui_smoke.py` | Demonstrate + smoke-test the **existing** Bio Lab UI: list models, run on-demand prediction for demo genes, validate the genome-view API response, pre-warm the prediction cache. **Base models only** (current state). | Working |
| 02 | `02_build_showcase_feature_cache.py` | **Phase A** of the meta→UI plan. Build per-gene dense multimodal feature `.npz` (reusing the training cache builder) for showcase genes, then verify M1-S end-to-end (recovers base-model misses). Feeds the meta overlay. Full RBP by default (all eCLIP evidence, resolved internally — no flags). | Working |
| 03 | `03_cryptic_site_detection.py` | Probe base vs M1-S vs M2-S at ALS TDP-43 cryptic splice sites (STMN2/UNC13A) vs canonical (positive control). True OOD test of whether the meta layer flags cryptic isoforms the base model misses. | Working |
| 04 | `04_tdp43_ablation_counterfactual.py` | **M4 prototype.** TDP-43 (TARDBP) knockout counterfactual vs the RBP-aware `*_v3_neuronal` models. **Diagnosis:** M2-S now *detects* the UNC13A cryptic donor (P=0.517), but KO does not de-repress (wrong sign) — the model never sees the de-repressed state in training. Honest negative; solving M4 needs perturbation-paired labels. | Working (diagnostic) |
| 05 | `05_explainability_integrated_gradients.py` | captum Integrated Gradients at a cryptic site (default UNC13A donor): attributes M2-S's P(splice) to the one-hot DNA sequence vs the 9 multimodal channels (base scores held fixed). Writes `ig_summary.json` + `ig_attribution.npz` + saliency PNG. Answers "*why* does M2-S see it?" (≈57% sequence; `rbp_n_bound` top multimodal channel). | Working |
| 06 | `06_demo_synthesis.py` | **Milestone story board.** Reads the held-out result JSONs (no recompute) and renders a 4-panel figure + `DEMO_STORY.md`: M1-S canonical FN −93%, M2-S alt-site recall 6×, UNC13A cryptic detection, IG attribution. Runs locally in seconds. | Working |
| 07 | `07_warm_ui_cache.py` | **Phase E.** Warms a running Bio Lab server's caches for the showcase genes (base + each meta overlay) so the live demo's first click is instant; prints base→meta FN/FP per gene (end-to-end smoke test of the meta overlay). | Working |
| — | `als_cryptic_sites.py` | Curated, splice-site-aligned GRCh38 cryptic-exon coordinates for STMN2/UNC13A (source of truth; emits corrected BED/TSV). | Working |
| — | `fetch_encori_tardbp.py` | Pull **neuronal** TDP-43 (TARDBP) CLIP peaks from ENCORI/starBase (hg38, SH-SY5Y/H9) into the eCLIP schema. The all-cell-line RBP union (ENCODE K562/HepG2 ∪ neuronal SH-SY5Y/H9) is now the **default** RBP source, resolved internally by `DenseFeatureExtractor` — no per-script flags. | Working |

## Plan & current state

- Roadmap: [`dev/meta_layer/UI_integration/m1s_m2s_ui_integration_plan.md`](../../dev/meta_layer/UI_integration/m1s_m2s_ui_integration_plan.md)
- **Phases A–E DONE (2026-05-23): the genome view is now meta-capable.** The
  genome endpoint takes an optional `meta` param
  (`/api/genome/{gene}/predict?model=openspliceai&meta=m1s_v3_neuronal&threshold=T`)
  and returns a **base-vs-meta overlay** — base donor/acceptor (the OpenSpliceAI
  scores the meta layer refines) plus the meta layer's donor/acceptor + its own
  TP/FP/FN at the same positions. The genome view adds a **Meta overlay**
  dropdown, dashed-green meta lines, open-diamond meta markers, and dual
  base/meta TP-FP-FN counts. Verified live: UNC13A base FN 4 → M2-S 0.
  - Backend: `meta_model_cache.py` (B), `meta_inference.py` (C),
    `genome_predict` `meta` branch + `GenomeResponse.meta_*` (D),
    `07_warm_ui_cache.py` (E). Meta models registered in `settings.yaml`
    `meta_models:`.
  - The meta overlay sources `gene_data` from the Phase-A `.npz` cache, so it is
    **instant for warmed showcase genes** and streams bigWigs (slow, first-load)
    only for novel genes — warm them with `02` / `07` before a live demo.

## Running

```bash
# Use the env python directly if `conda run` errors with "__conda_exe: permission denied":
PY=~/miniforge3/envs/agentic-spliceai/bin/python

# 1. Exercise the existing Bio UI (auto-starts a temporary server, runs, tears down):
$PY examples/UI_integration/01_bio_ui_smoke.py --start-server

#    …or against an already-running server (start it once, keep it up for the demo):
$PY -m server.bio.app          # -> http://localhost:8005
$PY examples/UI_integration/01_bio_ui_smoke.py

# 2. Build + verify the meta-model feature cache for the showcase genes (Phase A):
$PY examples/UI_integration/02_build_showcase_feature_cache.py

# 5. Explain why M2-S sees the UNC13A cryptic donor (Integrated Gradients):
$PY examples/UI_integration/05_explainability_integrated_gradients.py \
    --gene UNC13A --kind donor --model-dir output/meta_layer/m2s_v3_neuronal

# 6. Render the demo story board from held-out result JSONs (runs locally, no recompute):
$PY examples/UI_integration/06_demo_synthesis.py
#    -> output/meta_layer/demo_synthesis/{demo_story.png, DEMO_STORY.md}

# 7. Meta overlay in the live UI: start the server, then warm + smoke-test the
#    base-vs-meta overlay for the showcase genes (browse http://localhost:8005/genome/UNC13A):
$PY -m server.bio.app &                         # port 8005
$PY examples/UI_integration/07_warm_ui_cache.py
```

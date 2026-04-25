# Applications

Public ledger of matured application bundles for Agentic-SpliceAI.

An **application** is a refined, packaged bundle of use cases that
represents a user-facing functionality aligned with project goals. Each
entry below curates a subset of example scripts under `examples/` into a
coherent unit, declares the `src/` surface it depends on, and records its
current maturity.

For the biology and translational-impact narrative (oncology, clinical
VUS, neurology, drug development, biomarkers), see
[use_cases.md](use_cases.md).

---

## Project goals

Agentic-SpliceAI's applications serve a hierarchy of goals:

1. **Adaptive splice prediction** — going beyond canonical annotations
   with context-aware models
2. **Novel isoform discovery** — identifying the 90% of splice sites
   outside MANE/RefSeq
3. **Drug target identification** — translating isoform discoveries into
   therapeutically actionable targets

Each application below declares which goal(s) it serves.

---

## Maturity tiers

Applications live in one of four tiers. For the full definitions,
graduation signals, and demotion triggers, see the methodology doc at
[`dev/system_design/maturity_lifecycle.md`](../../dev/system_design/maturity_lifecycle.md).

| Tier | Meaning |
|------|---------|
| **Incubating** | 1-2 scripts, proof of concept; no ledger entry typically |
| **Active** | Multi-script workflow, results folder, benchmarks; ledger entry exists |
| **Mature** | Stable driver, reproducible benchmarks, downstream consumers |
| **Experimental** | Sub-project with its own lifecycle (e.g., foundation models) |
| **Product** | Deployable commitment (see [../products/](../products/) — empty) |

---

## Ledger

| # | Application | Goals served | Tier | Driving examples | Spec |
|---|-------------|--------------|------|------------------|------|
| 1 | Canonical Splice Prediction | Adaptive prediction | **Mature** | [`examples/base_layer/`](../../examples/base_layer/) | [canonical_splice_prediction/](canonical_splice_prediction/README.md) |
| 2 | Adaptive Splice Prediction (M1/M2) | Adaptive prediction | **Active** | [`examples/meta_layer/`](../../examples/meta_layer/) | [adaptive_splice_prediction/](adaptive_splice_prediction/README.md) |
| 3 | Multimodal Feature Engineering | Cross-cutting | **Active** | [`examples/features/`](../../examples/features/) | [multimodal_features/](multimodal_features/README.md) |
| 4 | Genomic Data Preparation | Cross-cutting | **Active** | [`examples/data_preparation/`](../../examples/data_preparation/) | [genomic_data_preparation/](genomic_data_preparation/README.md) |
| 5 | Variant Effect Analysis (M4) | Adaptive + drug target | **Active** | [`examples/variant_analysis/`](../../examples/variant_analysis/) | [variant_analysis/](variant_analysis/README.md) |
| 6 | Bioinformatics Lab UI | All goals | **Mature** | [`server/bio/`](../../server/bio/), [`notebooks/bioinfo_ui/`](../../notebooks/bioinfo_ui/) | [bioinformatics_lab_ui/](bioinformatics_lab_ui/README.md) |
| 7 | Novel Isoform Discovery (M3) | Isoform discovery | **Incubating** | (planned) | [novel_isoform_discovery/](novel_isoform_discovery/README.md) |
| 8 | Agentic Validation | All goals | **Incubating** | [`examples/agentic_layer/`](../../examples/agentic_layer/) | [agentic_validation/](agentic_validation/README.md) |
| 9 | Foundation Model Predictors | Adaptive prediction | **Experimental** | [`examples/foundation_models/`](../../examples/foundation_models/) | [foundation_model_predictors/](foundation_model_predictors/README.md) |

**Reading the ledger**: applications are named by **user-facing
functionality**, not by `examples/<topic>/` folder name. One topic can
seed multiple applications; one application can pull from multiple topics.

---

## How to read a spec

Each application spec (`docs/applications/<name>/README.md`) follows the
same structure, adapted from the template at
[_template.md](_template.md):

- **Problem** — what biological or computational question is being
  answered?
- **User-facing functionality** — what does this application let a user
  do?
- **Goals served** — which of the three project goals does this advance?
- **Driving examples** — the curated list of `examples/<topic>/*.py`
  scripts that demonstrate the application
- **`src/` surface** — the library modules the application relies on
- **Evaluation** — benchmarks, baselines, metrics; links to results
  folders
- **Maturity tier** — current tier with the signals that support it
- **Graduation signals** — what would move this to the next tier
- **Known limitations** — scope boundaries, failure modes

---

## Adding a new application

An application is created when an `examples/<topic>/` reaches late-Active
maturity (see
[`dev/system_design/maturity_lifecycle.md`](../../dev/system_design/maturity_lifecycle.md)):
the topic has multiple scripts, reproducible benchmarks, a canonical
driver, and a clear user-facing functionality.

Process:

1. Draft an internal harvest note at `dev/applications/<name>-harvest.md`
   scoping the application
2. Copy [_template.md](_template.md) to
   `docs/applications/<name>/README.md`
3. Fill in each section, linking to driving examples, `src/` modules, and
   results
4. Add a row to the ledger table above
5. Cross-link from any related tutorials under `docs/tutorials/`

---

## Promotion to product

No application is currently being promoted. Product promotion requires
all six criteria under
[`dev/system_design/maturity_lifecycle.md#tier-4--product`](../../dev/system_design/maturity_lifecycle.md)
to be demonstrably met, and a deliberate decision to take on maintenance.

The products catalog lives at [../products/](../products/) and is
currently empty.

---

## See also

- [use_cases.md](use_cases.md) — biology and translational-impact
  narrative (oncology, clinical VUS, neurology, drug dev, biomarkers)
- [`dev/system_design/`](../../dev/system_design/) — portable R&D
  methodology behind this structure
- [`dev/applications/`](../../dev/applications/) — internal WIP notes on
  applications
- [../products/README.md](../products/README.md) — future product catalog
- [../ROADMAP.md](../ROADMAP.md) — phase-level project status

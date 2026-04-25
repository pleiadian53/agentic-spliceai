# <Application Name>

<!--
Copy this file to docs/applications/<name>/README.md and fill in each
section. Keep it concise — this is a ledger entry, not a tutorial. Link
out to existing docs, examples, and results folders rather than
duplicating them.
-->

**Goals served**: <one or more of: adaptive splice prediction / isoform
discovery / drug target identification / cross-cutting>

**Tier**: <Incubating / Active / Mature / Experimental>

**Last updated**: <YYYY-MM>

---

## Problem

<2-4 sentences: the biological or computational question this application
answers. Why does it matter? Who is the intended user?>

## User-facing functionality

<What does this application let a user do, in concrete terms? Bullet list
of capabilities. Be specific — "run X on input Y and get Z" rather than
"analyze splicing".>

- ...
- ...

## Driving examples

The curated list of `examples/` scripts that constitute this application.
Each entry links to the script and names its role in the workflow.

- [`examples/<topic>/NN_script.py`](../../../examples/<topic>/NN_script.py) —
  <one-line purpose>
- ...

Supporting notebooks (if any):

- [`notebooks/<topic>/XX_*.ipynb`](../../../notebooks/<topic>/XX_*.ipynb) —
  <intuition / tutorial>

## `src/` surface

Library modules the application relies on. Keep this list minimal and
accurate — it should reflect real imports in the driving examples.

- `agentic_spliceai.<module>.<function>` — <what it provides>
- ...

Configuration profiles (if YAML-driven):

- [`path/to/config.yaml`](../../../path/to/config.yaml) — <profile purpose>

## Evaluation

Benchmarks, datasets, baselines, and key metrics. Link to result files
rather than copying numbers.

- **Dataset(s)**: <e.g., ClinVar splice-filtered, MutSpliceDB>
- **Baselines compared**: <SpliceAI, prior version, deterministic oracle>
- **Key metrics**: <PR-AUC, consequence concordance, wall-clock>
- **Results**: [`examples/<topic>/results/*.md`](../../../examples/<topic>/results/)

## Maturity tier and signals

**Current tier**: <Incubating / Active / Mature / Experimental>

**Signals supporting the tier**:

- <e.g., "11 scripts with stable args across 3 months">
- <e.g., "benchmark PR-AUC 0.92 vs SpliceAI 0.75 on ClinVar splice SNVs">
- <e.g., "depended on by 2 other applications">

## Graduation signals

**To advance to <next tier>, the application needs**:

- ...
- ...

## Known limitations

- <scope boundary or failure mode>
- ...

## Related

- [Roadmap entry](../../ROADMAP.md#<anchor>)
- Related applications: [other app](../other_app/README.md)
- Tutorials: [`docs/tutorials/<guide>.md`](../../tutorials/)
- Methodology: [`dev/system_design/maturity_lifecycle.md`](../../../dev/system_design/maturity_lifecycle.md)

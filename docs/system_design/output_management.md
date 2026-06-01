# Output & Results Management

A consistent way to put produced artifacts on disk, surface them via a
registry, and keep track of what's active vs superseded — so finding the
right model, eval, or audit never depends on memory.

This page is the **rulebook**. The live state lives in two files:

- [`src/agentic_spliceai/splice_engine/config/settings.yaml`](../../src/agentic_spliceai/splice_engine/config/settings.yaml)
  — authoritative for the *runtime* (which base models / meta models the
  system loads). Read by `splice_engine/resources/`.
- `output/REGISTRY.md` — authoritative for *discovery* (everything physically
  in `output/`, regardless of whether it's promoted). Tracked locally; the
  rules below define what should be in it.

## Two kinds of "produced" things

The project distinguishes **outputs** from **curated results**:

| | Outputs | Curated results |
|---|---|---|
| **Where** | `output/<topic>/<artifact>/` | `examples/<topic>/results/` |
| **What** | Raw artifacts: model checkpoints, eval JSON, predictions, caches | Narrative summaries (markdown) + small key reference files |
| **Size** | Any (can be large) | Small (intentionally) |
| **Gitignored?** | `output/` is gitignored | Tracked in git |
| **Audience** | Code consumers, the registry | Humans reading the project |
| **Lifecycle** | Long-lived but rotatable | Permanent record of what was found |

An "output" is what a script produced; a "curated result" is the
human-facing story about it. A given finding usually has both.

## The `output/<topic>/<artifact>/` convention

- **Topic** = a major workstream. Current topics: `meta_layer`,
  `m4_benchmarks`, `exon_classifier`, `fm_scalars`, `gpu_runs`,
  `splice_classifier`, `bio_cache`. New workstream → new topic dir.
- **Artifact name** must be self-describing. Good: `m1s_v4_cleanannot`,
  `mutsplicedb_m2s_v2_r50`, `clinvar_splice_m1s_v2_r100`. Bad:
  `experiment2`, `final_run`, `test`.
- For dated runs that are 1-of-N replicates, suffix the date:
  `openspliceai_GRCh38_20260303_141338/`.
- Inside an artifact dir, conventional files: `best.pt`, `config.pt`,
  `final.pt`, `eval_results.json`, `train.log` (training runs);
  `benchmark_metrics.json`, `delta_scores.json` (eval runs); a small
  `MANIFEST.md` is encouraged when the layout isn't obvious.

## Status taxonomy

A **tag**, never a path component. Status changes never move files.

| Tag | Meaning |
|---|---|
| `active` | Current best of its class; referenced by code/docs; promoted in `settings.yaml` (for models). |
| `baseline` | Kept as a historical reference for comparison (e.g. v2 vs v4). Not promoted. |
| `experimental` | Trained / produced but not yet plumbed through a runtime protocol. May graduate to `active` or be dropped. |
| `archived` | Kept for reproducibility; not referenced by current code/docs. |
| `placeholder` | Empty / pointer dir; the underlying work moved elsewhere (e.g. to a sibling project). |

`stale` is **not** a tag — it's a deletion candidate. If something is
stale, delete it (and remove its row from the registry).

## The two-layer registry

| Layer | File | What it tracks | Source of truth for |
|---|---|---|---|
| Runtime | `settings.yaml` | Only **active** models (`base_models:` + `meta_models:` blocks) | What the system loads at runtime |
| Discovery | `output/REGISTRY.md` | **Everything** in `output/` with a status tag | What's on disk + why |

No duplication: an `active` model appears in both, with the registry row
pointing at the `settings.yaml` block. Baselines / experimental /
archived artifacts only appear in the registry.

### Why not a single source

The runtime config is consumed by code (`resources/get_meta_model_config()`
returns a dict that's loaded as a model). It should stay small and
canonical. Bloating it with stale history would make the resolution layer
brittle and slow.

The discovery registry is for humans (and future-you in another session)
to find things without remembering. It naturally grows; it doesn't need
to be parseable by the runtime.

## Path resolution — never hardcode

Modules and scripts MUST resolve output paths through the resource
manager, not by string-concatenating. Two main entry points:

```python
from agentic_spliceai.splice_engine.resources import (
    get_model_resources,   # base models: spliceai, openspliceai, ...
    get_meta_model_config, # meta models: m1s_v4_cleanannot, ...
)

# Base model resources (build, GTF, FASTA, weights dir, etc.):
res = get_model_resources("openspliceai")
gtf = res.get_registry().get_gtf_path()

# Meta model config (dir, name, notes, ...):
cfg = get_meta_model_config("m1s_v4_cleanannot")
checkpoint = Path(cfg["dir"]) / "best.pt"
```

Anything that breaks if you `mv output/meta_layer/m1s_v4_cleanannot ...`
is a violation. If the dir name needs to change, edit `settings.yaml`.

## Common operations

### Adding a new artifact

1. Place it under `output/<topic>/<self_describing_name>/`.
2. Add a row to `output/REGISTRY.md` (topic section, with status tag,
   produced-by script/command, one-line notes).
3. If it's a model that downstream code should load, also add it to
   `settings.yaml` under `base_models:` or `meta_models:`. Set the
   registry status to `active`.

### Promoting a model

1. Add (or update) the entry in `settings.yaml meta_models:`.
2. Update the registry row's status to `active`.
3. Update the previous canonical's status to `baseline` (if you want to
   keep it for comparison) or delete it.
4. **No file moves.** Anything depending on `get_meta_model_config()`
   keeps working.

### Demoting / superseding

1. Edit the registry row's `status` from `active` to `baseline` / `archived`.
2. Remove the entry from `settings.yaml` (if it was there).
3. Add a `superseded_by` note pointing at the replacement.
4. **No file moves.**

### Cleanup heuristic

Anything tagged `archived` and not load-bearing for ~2+ sessions is a
deletion candidate. Drop the directory + its registry row in one go.
Before deleting, `grep -r <artifact_name>` to confirm no live references.

`baseline` rows stay indefinitely — they exist precisely because we want
to compare against them.

## Discovery — common queries

| Question | Where to look |
|---|---|
| "Where is M1-S?" | `grep m1s output/REGISTRY.md` → path + status. In code: `get_meta_model_config('m1s_v4_cleanannot')`. |
| "What's the v2 baseline for comparison?" | Registry → rows tagged `baseline`. |
| "What experiments did we run on MutSpliceDB?" | Registry → rows under `output/m4_benchmarks/` + `examples/variant_analysis/results/`. |
| "Why does this big `splice_classifier/` dir exist?" | Registry row → tag `experimental`, notes link to a BACKLOG item explaining the integration gap. |
| "What's in a sibling project's domain?" | Registry placeholders → e.g. `output/biomol_design/README.md` points at `protein-ml-lab`. |

## Curated results — `examples/<topic>/results/`

This is the **human-facing narrative layer**. One file per
result-worth-citing. Each file has:

- A title that names the experiment + model version
- A status banner near the top (active / superseded / re-run pending)
- Headline numbers (the 2–4 that matter)
- A pointer to the underlying output dir
- Honest caveats and what was *not* tested

Example: [`examples/variant_analysis/results/m4_benchmark_sweep.md`](../../examples/variant_analysis/results/m4_benchmark_sweep.md)
documents the April 2026 MutSpliceDB + ClinVar sweep, with a 2026-05-31
validation banner at the top pointing at the v4 re-run results.

When a result is re-run with newer models, **don't overwrite the
narrative file** — add a `SUPERSEDED` banner with a pointer to the new
result. The history of how numbers evolved is itself useful.

## Conventions for new findings

When you produce a new finding worth keeping:

1. **Output goes in `output/<topic>/<name>/`** (raw artifacts).
2. **Registry row** added (status, produced-by, notes).
3. **Curated result** added at `examples/<topic>/results/<name>.md` if
   it's a publication-worthy finding (most metric/benchmark results
   qualify; sanity checks usually don't).
4. **If a model is promoted**: `settings.yaml` edit + previous version
   demoted in the registry.
5. **If a script + run produced something interesting that should
   become a reusable utility**: extract to `src/`, leave the
   experimental script as a use-case-driven example in `examples/`.

## What's NOT covered here

This doc is about **artifacts**. Adjacent concerns documented elsewhere:

- **Code organization** ([`../architecture/PACKAGE_ORGANIZATION.md`](../architecture/PACKAGE_ORGANIZATION.md))
- **Directory layout** ([`../architecture/STRUCTURE.md`](../architecture/STRUCTURE.md))
- **Resource resolution** for genomic data inputs (FASTA, GTF, etc.):
  [`resource_management.md`](resource_management.md)
- **Configuration**: [`configuration_system.md`](configuration_system.md)
- **Data inputs** — large reference data (FASTA, GTF, bigWigs) lives
  under `data/<source>/<build>/` with its own conventions.
- **Sessions / tasks** — private development log lives outside the
  repo's public docs.

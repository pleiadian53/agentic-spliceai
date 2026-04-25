# Products

Deployable, committed artifacts built on Agentic-SpliceAI.

A **product** is an application that has graduated from
`docs/applications/<name>/` to this tier. Products are **commitments to
users**: stable API, versioned checkpoints, benchmarked performance,
inference-path tests, documented limitations.

**Current status**: no products yet. Project focus is
[application maturity](../applications/README.md).

---

## Applications vs Products

| | [Applications](../applications/) | Products (here) |
|---|---|---|
| **Claim** | "Here's an approach you could use" | "Here's a thing you can run" |
| **Code location** | `examples/<topic>/` + curated in `docs/applications/<name>/` | `src/agentic_spliceai/applications/<name>/` |
| **API** | Whatever fits the exploration | Stable public interface |
| **Tests** | Informal | Inference path covered |
| **Artifacts** | Optional | Versioned checkpoints with metadata |
| **Baseline comparison** | Nice to have | Required |
| **Deployability** | Not expected | Inference without training infrastructure |
| **Maintenance** | Best-effort | Ongoing commitment |

Applications are a bet on methodology. Products are a commitment to users.

For the underlying methodology and tier definitions, see
[`dev/system_design/maturity_lifecycle.md`](../../dev/system_design/maturity_lifecycle.md).

---

## Promotion criteria

To graduate from `docs/applications/<name>/` to `docs/products/<name>/`,
the work must satisfy **all six** of the following. "Mostly met" is not
sufficient — each criterion is a binary gate.

1. **Code maturity** — implementation lives under
   `src/agentic_spliceai/applications/<name>/` (not `examples/`), with a
   stable public API. Signature changes constitute a breaking change.
2. **Evaluation** — benchmarked against at least one published baseline
   (e.g., SpliceAI on ClinVar, MutSpliceDB consequence concordance, GTEx
   junction recall). Results reproducible from a clean environment with
   the documented command.
3. **Testability** — test suite in `tests/` covers the inference path at
   minimum. Training path coverage is a plus but not required for
   promotion.
4. **Deployability** — a CLI entry point via `pyproject.toml
   [project.scripts]` (e.g., `agentic-spliceai-variant`) or a library-level
   public function with stable signature. Inference must run without
   requiring training-time infrastructure (no SkyPilot invocation inside
   `predict()`; no Evo2 40B dependency for an M1-S product).
5. **Artifacts** — trained checkpoints versioned with metadata: seed,
   data version/hash, training config, performance on the benchmark.
   Artifacts live on a network volume (RunPod volume, HuggingFace Hub)
   or public host, not in git.
6. **Documentation** — `docs/products/<name>/README.md` covers:
   - What the product does and does not do
   - Installation and inference quickstart
   - Expected performance on the benchmark, with confidence intervals
   - Known limitations and failure modes
   - A runnable notebook under `notebooks/<topic>/` showing end-to-end
     use on a new input

---

## Current products

_(none yet)_

The closest candidates, by current application maturity, are:

| Candidate | Current tier | Gap from product |
|---|---|---|
| [Canonical Splice Prediction](../applications/canonical_splice_prediction/README.md) | Mature application | Artifact versioning, inference-path tests, product-level docs |
| [Bioinformatics Lab UI](../applications/bioinformatics_lab_ui/README.md) | Mature application | Stable API contract, deployment guide, external hosting, inference-path tests |
| [Variant Effect Analysis](../applications/variant_analysis/README.md) | Active application | Phase 8.3 clinical head, CLI entry point, versioned checkpoints, tests |

None of these are being actively prepared for promotion. Promotion
requires a deliberate decision and a commitment to ongoing maintenance —
see [`dev/system_design/maturity_lifecycle.md`](../../dev/system_design/maturity_lifecycle.md).

---

## Demotion

Products can be demoted back to applications when they regress — e.g., an
upstream dataset changes, a baseline comparison becomes invalid, or a
dependency breaks the inference path. Demotion is healthier than
silently letting a stale product live in the product tier. See
[`dev/system_design/maturity_lifecycle.md#demotion`](../../dev/system_design/maturity_lifecycle.md) for
the full demotion protocol.

---

## Adding a new product

When an application meets all six criteria:

1. Confirm each criterion is met (binary gate, not "mostly").
2. Draft an internal promotion note at `dev/products/<name>-promotion.md`
   working through each criterion with evidence.
3. Move implementation from `examples/<topic>/` and/or library modules
   into `src/agentic_spliceai/applications/<name>/`. Keep a stable public
   API.
4. Add CLI entry point to `pyproject.toml [project.scripts]`.
5. Add inference-path test coverage under `tests/`.
6. Version trained checkpoints with a manifest file (seed, data hash,
   config, metrics).
7. Create `docs/products/<name>/README.md` using the structure above.
8. Add a one-line entry to "Current products" in this file.
9. Update the predecessor application spec to point at the product
   (`docs/applications/<name>/README.md` → "Graduated to product, see
   `docs/products/<name>/`").
10. Announce in a session summary under `dev/sessions/YYYY-MM-DD_<name>-graduated.md`.

---

## See also

- [../applications/README.md](../applications/README.md) — public application ledger
- [`dev/products/README.md`](../../dev/products/README.md) — internal product tracking
- [`dev/system_design/maturity_lifecycle.md`](../../dev/system_design/maturity_lifecycle.md) — methodology
- [../ROADMAP.md](../ROADMAP.md) — phase-level project status

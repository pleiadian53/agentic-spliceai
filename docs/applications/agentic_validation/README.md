# Agentic Validation

**Goals served**: all goals (orthogonal evidence layer)

**Tier**: Incubating

**Last updated**: 2026-04

---

## Problem

Splice-site predictions benefit from cross-referencing against biological
knowledge that isn't captured in multimodal features: published
literature, expression patterns across tissues and disease states,
clinical variant databases, cross-species conservation context,
structural predictions. Pulling this evidence manually doesn't scale;
LLM-powered agents can orchestrate multi-source lookups and synthesize
structured evidence per candidate.

The goal is an agentic validation layer that takes a set of candidate
splice sites (from M1/M2/M3/M4) and produces evidence-annotated reports,
with an eventual self-improvement loop that feeds validation signals
back into the meta layer.

## User-facing functionality (planned)

- Submit a set of candidate sites or isoforms; receive a structured
  report combining:
  - **Literature agent**: PubMed, arXiv, Europe PMC lookups
  - **Expression agent**: GTEx, TCGA, ENCODE tissue-specific expression
  - **Clinical agent**: ClinVar, COSMIC, SpliceVarDB
  - **Conservation agent**: PhyloP, PhastCons cross-species context
  - **Structural agent** (later): AlphaFold, Foldseek structure prediction
- Generate PDF research reports via Nexus research agent
- Self-improvement feedback loop — validation outcomes update meta-layer priors

## Driving examples

- [`examples/agentic_layer/quick_start.py`](../../../examples/agentic_layer/quick_start.py) — minimal agent demo
- [`examples/agentic_layer/analyze_splice_sites.py`](../../../examples/agentic_layer/analyze_splice_sites.py) — splice-site analysis with LLM
- [`examples/agentic_layer/docs/`](../../../examples/agentic_layer/docs/) — design notes

Supporting infrastructure:

- **Nexus research agent** (`src/nexus/`) — planner → researcher → writer → editor pipeline
- **CLI entry points** (already exist): `nexus`, `nexus-server`

## `src/` surface

- `nexus.agents.research.ResearchAgent` — research agent orchestrator
- `nexus.core.config.Config` — LLM provider configuration
- `agentic_spliceai.agents.*` — (planned) splice-specific agents
- `agentic_spliceai.analysis.*` — analysis template execution
- `agentic_spliceai.server.splice_service` — REST API (port 8004) for
  template-driven analysis

## Evaluation (planned)

- **Benchmark**: concordance between agent-produced clinical
  classifications and expert curation (ClinVar P/LP/B/LB)
- **Metric**: inter-agent agreement, citation precision, literature
  coverage
- **Human-in-the-loop**: initial validation requires expert review until
  the loop converges

## Maturity tier and signals

**Current tier**: Incubating

**Signals supporting the tier**:

- 2 example scripts (quick_start, analyze_splice_sites)
- Nexus agent available and stable (used independently for research
  report generation)
- Splice service REST API exists (`agentic-spliceai-server`)
- No splice-specific agents yet — primary gap

## Graduation signals

**To advance to Active, the application needs**:

- At least 2 splice-specific agents (e.g., Literature + Expression)
  implemented and demonstrated on a small candidate set
- Agent output schema stabilized (structured evidence records)
- Benchmark on a curated variant set (e.g., the 13 disease-gene variants
  from variant_analysis)
- Integration with one downstream application (e.g., Variant Effect
  Analysis producing candidates, agents producing validation)

## Known limitations (anticipated)

- LLM output requires careful grounding — hallucinated citations are the
  primary failure mode
- Rate limits on literature APIs will bottleneck large candidate sets
- Self-improvement feedback loop requires careful handling to avoid
  reward hacking or confirmation bias

## Related

- [Novel Isoform Discovery](../novel_isoform_discovery/README.md) — primary upstream source of candidates
- [Variant Effect Analysis](../variant_analysis/README.md) — provides variant-level candidates
- [Use cases: all clinical scenarios](../use_cases.md) — agentic validation feeds every translational scenario
- [Roadmap: Phase 7](../../ROADMAP.md)
- [Nexus Research Agent](../../tutorials/SPLICE_PREDICTION_GUIDE.md)

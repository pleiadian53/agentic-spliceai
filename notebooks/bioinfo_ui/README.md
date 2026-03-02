# Bioinformatics UI Notebooks

Educational notebooks demonstrating the AgenticSpliceAI Lab web UI and API.

## Available Notebooks

### 01_serpina1_copd_use_case/

**Splice Site Investigation: SERPINA1, UNC13A, STMN2** — Walk through a complete
splice site investigation using the bioinformatics UI API, covering COPD-related
(SERPINA1) and ALS-related (UNC13A, STMN2) genes.

- Discover models and browse the gene catalog
- Predict and visualize splice sites
- Analyze FP/FN errors with clinical context
- Explore threshold sensitivity
- Compare model performance

**Prerequisites**: Server running on `localhost:8005`

## Learning Objectives

1. How to interact with the AgenticSpliceAI Lab API programmatically
2. How to interpret splice site predictions in a clinical context
3. How to analyze prediction errors (FP/FN) and their biological meaning
4. How to compare models and tune thresholds for different use cases

## See Also

- **Web UI**: http://localhost:8005 (Gene Browser, Genome View, Metrics Dashboard)
- **Server code**: [`../../server/bio/`](../../server/bio/)
- **Examples**: [`../../examples/base_layer/`](../../examples/base_layer/)

---

Last Updated: March 2026

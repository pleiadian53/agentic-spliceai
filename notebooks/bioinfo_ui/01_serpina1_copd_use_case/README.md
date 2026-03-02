# Splice Site Investigation: SERPINA1 (COPD), UNC13A & STMN2 (ALS)

Demonstrates the AgenticSpliceAI Lab bioinformatics UI through an investigation
of clinically important genes:

- **SERPINA1** — alpha-1 antitrypsin deficiency (AATD) and COPD
- **UNC13A** — cryptic exon inclusion in ALS/FTD
- **STMN2** — TDP-43-dependent cryptic splicing in ALS

## Quick Start

1. Start the server:
   ```bash
   cd /path/to/agentic-spliceai
   conda run -n agentic-spliceai python -m server.bio.app
   ```
2. Open `01_serpina1_copd_use_case.ipynb` in Jupyter
3. Run all cells

## What You'll Learn

- Using the Gene Browser API to search for clinically relevant genes
- Generating and interpreting the Genome View for SERPINA1, UNC13A, and STMN2
- Analyzing false positive and false negative splice site predictions
- Understanding threshold sensitivity for clinical vs research settings
- Comparing model performance across evaluation runs

## Prerequisites

- AgenticSpliceAI Lab server running on `localhost:8005`
- Python packages: `requests`, `plotly`, `pandas`, `numpy`

## Supplements

- [`supplements/01_serpina1_biology.md`](supplements/01_serpina1_biology.md) — Background on SERPINA1/AAT/COPD and UNC13A/STMN2/ALS

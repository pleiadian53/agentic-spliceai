"""Verify the code-as-plan chart pipeline end-to-end without an LLM.

`agentic_layer.planning.code_as_plan` and `agentic_layer.execution.code_executor`
work together: the planner generates Python chart code; the executor
runs it in a controlled namespace. This script feeds the executor a
hardcoded plan (skipping the LLM stage) and asserts a chart image is
produced. With --live, it goes through the full LLM-driven planner.

Run:
    mamba run -n agentic-spliceai python examples/agentic_layer/02_chart_pipeline.py
    mamba run -n agentic-spliceai python examples/agentic_layer/02_chart_pipeline.py --live
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from agentic_spliceai.agentic_layer.data.data_access import DataFrameDataset
from agentic_spliceai.agentic_layer.execution.code_executor import execute_chart_code


HARDCODED_PLAN_CODE = """
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df['gene'], df['count'])
ax.set_title('Splice events per gene (synthetic)')
ax.set_xlabel('Gene')
ax.set_ylabel('Event count')
plt.tight_layout()
plt.savefig('{OUTPUT_PATH}')
plt.close()
"""


def test_executor_runs_chart_code() -> None:
    df = pd.DataFrame({
        "gene": ["BRCA1", "TP53", "EGFR", "MYC"],
        "count": [42, 67, 23, 88],
    })
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "chart.png"
        code = HARDCODED_PLAN_CODE.replace("{OUTPUT_PATH}", str(out_path))
        result = execute_chart_code(code, df)
        assert out_path.exists(), f"chart not produced at {out_path}"
        assert out_path.stat().st_size > 1000, "chart PNG is suspiciously small"
        print(f"[OK] executor ran chart code; PNG produced ({out_path.stat().st_size} bytes)")


def test_dataset_wraps_dataframe() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    ds = DataFrameDataset(df, name="smoke")
    schema = ds.get_schema_description()
    sample = ds.get_sample_data(rows=2)
    assert "Rows: 3" in schema or "rows" in schema.lower(), schema
    assert "x" in sample and "y" in sample, sample
    print("[OK] DataFrameDataset round-trips schema + samples")


def test_live_planner() -> None:
    """End-to-end: LLM planner generates code; executor runs it."""
    if "OPENAI_API_KEY" not in os.environ:
        print("[SKIP] live planner test — OPENAI_API_KEY not set")
        return
    from openai import OpenAI
    from agentic_spliceai.agentic_layer.planning.code_as_plan import generate_chart_code

    client = OpenAI()
    df = pd.DataFrame({
        "gene": ["BRCA1", "TP53"],
        "events": [42, 67],
    })
    ds = DataFrameDataset(df, name="splice_smoke")
    result = generate_chart_code(
        dataset=ds,
        user_request="Bar chart of events per gene. Headless backend. Save to /tmp/live_chart.png.",
        client=client,
        model="gpt-4o-mini",
    )
    assert result.get("code"), f"planner returned no code: {result!r}"
    print(f"[OK] live planner produced {len(result['code'])} bytes of chart code")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run LLM planner (needs OPENAI_API_KEY)")
    args = parser.parse_args()

    print("Chart-pipeline verification (planner + executor)")
    print("=" * 50)
    test_dataset_wraps_dataframe()
    test_executor_runs_chart_code()
    if args.live:
        test_live_planner()
    else:
        print("[SKIP] live planner call (pass --live to enable)")
    print("=" * 50)
    print("All checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

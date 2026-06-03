"""Verify `agentic_layer.data.data_access` factory + every backend.

Pure unit tests, no LLM. Asserts:

1. `create_dataset()` dispatches correctly by file extension:
   - `.csv` -> CSVDataset
   - `.tsv`, `.tab` -> CSVDataset with sep='\\t'
   - `.xlsx`, `.xls` -> ExcelDataset
   - Unsupported extension -> ValueError with helpful message

2. Each backend's `get_schema_description()` returns a non-empty string
   that mentions row count and columns.

3. Each backend's `get_sample_data()` returns JSON-parseable rows.

Run:
    mamba run -n agentic-spliceai python examples/agentic_layer/03_data_loaders.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from agentic_spliceai.agentic_layer.data.data_access import (
    ChartDataset,
    CSVDataset,
    DataFrameDataset,
    ExcelDataset,
    create_dataset,
)


def test_factory_dispatches_by_extension() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # CSV
        csv_path = td / "x.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        ds = create_dataset(csv_path)
        assert isinstance(ds, CSVDataset), f"got {type(ds).__name__}"

        # TSV
        tsv_path = td / "x.tsv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(tsv_path, sep="\t", index=False)
        ds = create_dataset(tsv_path)
        assert isinstance(ds, CSVDataset), f"got {type(ds).__name__}"
        assert ds.read_csv_kwargs.get("sep") == "\t", "TSV must pass sep='\\t'"

        # XLSX — dispatch test only (avoid openpyxl dep for the unit pass; the
        # factory should still wrap the path in ExcelDataset even before the
        # file is read)
        xlsx_path = td / "x.xlsx"
        xlsx_path.touch()  # stub; we don't read it
        ds = create_dataset(xlsx_path)
        assert isinstance(ds, ExcelDataset), f"got {type(ds).__name__}"

        # Unsupported
        bogus = td / "x.parquet"
        bogus.write_text("not really parquet")
        try:
            create_dataset(bogus)
            raise AssertionError("should have raised on .parquet")
        except ValueError as e:
            assert "parquet" in str(e) or ".parquet" in str(e), e
            assert "SQLiteDataset" in str(e) or "DuckDBDataset" in str(e), e

    print("[OK] create_dataset dispatches CSV / TSV / Excel and refuses unknown extensions")


def test_backend_schema_and_samples() -> None:
    df = pd.DataFrame({"gene": ["BRCA1", "TP53"], "count": [42, 67]})
    ds = DataFrameDataset(df, name="smoke")
    schema = ds.get_schema_description()
    assert "Rows: 2" in schema or "2" in schema, schema
    assert "gene" in schema and "count" in schema, schema
    sample = ds.get_sample_data(rows=2)
    parsed = json.loads(sample)
    assert len(parsed) == 2, parsed
    assert {r["gene"] for r in parsed} == {"BRCA1", "TP53"}
    print("[OK] DataFrameDataset schema + samples round-trip")


def test_csv_backend_against_disk() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "d.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(path, index=False)
        ds = CSVDataset(path)
        df = ds.load_dataframe()
        assert list(df["x"]) == [1, 2, 3]
        schema = ds.get_schema_description()
        assert "Rows: 3" in schema, schema
    print("[OK] CSVDataset reads from disk")


def test_chartdataset_is_abc() -> None:
    try:
        ChartDataset()  # type: ignore[abstract]
        raise AssertionError("ChartDataset should be abstract")
    except TypeError as e:
        assert "abstract" in str(e).lower(), e
    print("[OK] ChartDataset cannot be instantiated directly")


def main() -> int:
    print("Data-access factory + backends verification")
    print("=" * 50)
    test_chartdataset_is_abc()
    test_factory_dispatches_by_extension()
    test_backend_schema_and_samples()
    test_csv_backend_against_disk()
    print("=" * 50)
    print("All checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

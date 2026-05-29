"""CLI smoke tests for the application-level entry points.

Each test executes a CLI scenario via ``subprocess.run`` and asserts on
``returncode`` plus one stable substring of stdout/stderr. Output
reformatting won't flake the tests; a real regression breaks them.

The scenarios mirror the human-curated reference under
``dev/cli_reference/``. When you add a new row to a reference table,
add a matching scenario tuple here.

Run all CLI smoke tests::

    pytest -m cli_smoke -v

Skip slow / pod-only ones::

    pytest -m 'cli_smoke and not slow and not pod' -v

Run one CLI's worth::

    pytest -m cli_smoke -v tests/cli/test_smoke.py::TestIngest
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
CONDA_ENV = "agentic-spliceai"
DEFAULT_TIMEOUT = 60  # seconds for non-slow tests


@dataclass
class CliResult:
    """Outcome of a CLI subprocess run."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        """Stdout + stderr concatenated. Useful when output channel is
        implementation-defined (e.g., logger writes to stderr by default).
        """
        return f"{self.stdout}\n{self.stderr}"


def run_cli(
    cmd: List[str],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
) -> CliResult:
    """Run a CLI command via ``conda run -n agentic-spliceai`` and return
    its outcome. Wraps ``subprocess.run`` with sane defaults.
    """
    full = ["conda", "run", "-n", CONDA_ENV] + cmd
    proc = subprocess.run(
        full,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
        cwd=str(cwd or REPO_ROOT),
    )
    return CliResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _has_cli(name: str) -> bool:
    """True if ``name`` is reachable inside the conda env."""
    res = subprocess.run(
        ["conda", "run", "-n", CONDA_ENV, "which", name],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def _has_path(p: str) -> bool:
    return Path(p).exists()


# ---------------------------------------------------------------------------
# Module-level skip if conda env is unavailable
# ---------------------------------------------------------------------------


pytestmark = [
    pytest.mark.cli_smoke,
    pytest.mark.skipif(
        shutil.which("conda") is None,
        reason="conda not on PATH; CLI smoke tests need conda run -n agentic-spliceai",
    ),
]


# ---------------------------------------------------------------------------
# TestIngest — agentic-spliceai-ingest
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cli("agentic-spliceai-ingest"),
                    reason="agentic-spliceai-ingest not installed")
class TestIngest:
    """Smoke tests for ``agentic-spliceai-ingest``.

    Reference: ``dev/cli_reference/agentic-spliceai-ingest.md``.
    """

    # ------------------------- list-builds -------------------------------

    def test_list_builds_table(self) -> None:
        """§1.1 — table format shows all configured builds."""
        r = run_cli(["agentic-spliceai-ingest", "list-builds"])
        assert r.returncode == 0
        assert "spliceai" in r.combined
        assert "openspliceai" in r.combined

    def test_list_builds_json(self) -> None:
        """§1.2 — JSON format is parseable and includes every build."""
        import json

        r = run_cli(["agentic-spliceai-ingest", "list-builds", "--format", "json"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        names = set(data.keys())
        assert {"spliceai", "openspliceai"}.issubset(names)

    # ------------------------- status ------------------------------------

    @pytest.mark.bio
    def test_status_canonical_mane_grch38_ready(self) -> None:
        """§2.1 — production data/mane/GRCh38 should be Ready: True."""
        r = run_cli([
            "agentic-spliceai-ingest", "status",
            "--canonical", "--build", "GRCh38", "--annotation-source", "mane",
        ])
        assert r.returncode == 0
        assert "Ready: True" in r.combined

    @pytest.mark.bio
    def test_status_canonical_ensembl_grch38_ready(self) -> None:
        """§2.2 — Ensembl GRCh38 should be Ready: True after 2026-04-25 gap-fill."""
        r = run_cli([
            "agentic-spliceai-ingest", "status",
            "--canonical", "--build", "GRCh38", "--annotation-source", "ensembl",
        ])
        assert r.returncode == 0
        assert "Ready: True" in r.combined

    @pytest.mark.bio
    def test_status_canonical_grch37_ensembl_partial(self) -> None:
        """§2.3 — GRCh37/ensembl: chromosome_split missing → exit 1."""
        r = run_cli([
            "agentic-spliceai-ingest", "status",
            "--canonical", "--build", "GRCh37", "--annotation-source", "ensembl",
        ])
        assert r.returncode == 1
        assert "Ready: False" in r.combined
        assert "chromosome_split" in r.combined

    def test_status_unconfigured_build(self) -> None:
        """§2.4 — T2T-CHM13 not configured → exit 2 with helpful message."""
        r = run_cli([
            "agentic-spliceai-ingest", "status",
            "--canonical", "--build", "T2T-CHM13", "--annotation-source", "mane",
        ])
        assert r.returncode == 2
        assert "Failed to resolve canonical dir" in r.combined

    def test_status_empty_output_dir(self, tmp_path: Path) -> None:
        """§2.5 — empty output dir reports manifest absent + all missing."""
        empty = tmp_path / "empty"
        empty.mkdir()
        r = run_cli([
            "agentic-spliceai-ingest", "status", "--output-dir", str(empty),
        ])
        assert r.returncode == 1
        assert "Manifest:   absent" in r.combined
        assert "Ready: False" in r.combined

    @pytest.mark.bio
    def test_status_json_format(self) -> None:
        """§2.6 — JSON output is valid + has the expected schema."""
        import json

        r = run_cli([
            "agentic-spliceai-ingest", "status",
            "--canonical", "--build", "GRCh38", "--annotation-source", "mane",
            "--format", "json",
        ])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["ready"] is True
        assert "artifacts" in data
        assert {"gene_features", "splice_sites", "chromosome_split"}.issubset(
            data["artifacts"].keys()
        )

    def test_status_no_args(self) -> None:
        """§2.7 — bare `status` is rejected by argparse."""
        r = run_cli(["agentic-spliceai-ingest", "status"])
        assert r.returncode == 2
        assert "required" in r.stderr.lower() or "required" in r.stdout.lower()

    # ------------------------- prepare -----------------------------------

    @pytest.mark.bio
    def test_prepare_dry_run(self) -> None:
        """§3.1 — dry-run prints plan, does not write."""
        r = run_cli([
            "agentic-spliceai-ingest", "prepare",
            "--inplace", "--build", "GRCh38", "--annotation-source", "mane",
            "--dry-run",
        ])
        assert r.returncode == 0
        assert "Dry run" in r.combined
        assert "Planned steps" in r.combined

    def test_prepare_throwaway_chromosome_split(self, tmp_path: Path) -> None:
        """§3.2 — throwaway --output-dir, single step, no hash."""
        out = tmp_path / "ingest_smoke"
        r = run_cli([
            "agentic-spliceai-ingest", "prepare",
            "--output-dir", str(out),
            "--build", "GRCh38", "--annotation-source", "mane",
            "--only-steps", "chromosome_split",
            "--no-hash",
        ])
        assert r.returncode == 0
        assert "Overall success: True" in r.combined
        assert (out / "chromosome_split.json").exists()
        assert (out / "ingest_manifest.json").exists()

    def test_prepare_no_dest_args(self) -> None:
        """§3.6 — neither --output-dir nor --inplace → argparse error."""
        r = run_cli([
            "agentic-spliceai-ingest", "prepare",
            "--build", "GRCh38", "--annotation-source", "mane",
        ])
        assert r.returncode == 2
        assert "required" in r.combined.lower()

    def test_prepare_unconfigured_build(self, tmp_path: Path) -> None:
        """§3.7 — T2T-CHM13 with --inplace cannot resolve canonical dir."""
        r = run_cli([
            "agentic-spliceai-ingest", "prepare",
            "--inplace",
            "--build", "T2T-CHM13", "--annotation-source", "mane",
        ])
        assert r.returncode == 2
        assert "Failed to resolve canonical dir" in r.combined

    def test_prepare_both_dest_args(self, tmp_path: Path) -> None:
        """§3.8 — --output-dir AND --inplace are mutually exclusive."""
        r = run_cli([
            "agentic-spliceai-ingest", "prepare",
            "--output-dir", str(tmp_path / "x"),
            "--inplace",
            "--build", "GRCh38", "--annotation-source", "mane",
        ])
        assert r.returncode == 2
        assert "not allowed" in r.combined.lower()

    # ------------------------- validate ----------------------------------

    @pytest.mark.bio
    def test_validate_mane_grch38(self, tmp_path: Path) -> None:
        """§4.1 — MANE/GRCh38 GTF parses + validates."""
        r = run_cli(
            [
                "agentic-spliceai-ingest", "validate",
                "--build", "GRCh38", "--annotation-source", "mane",
                "--output-dir", str(tmp_path),
            ],
            timeout=120,
        )
        assert r.returncode == 0
        assert "validate: ok" in r.combined

    @pytest.mark.bio
    def test_validate_ensembl_grch38(self, tmp_path: Path) -> None:
        """§4.2 — Ensembl/GRCh38 GTF (63K genes) parses + validates."""
        r = run_cli(
            [
                "agentic-spliceai-ingest", "validate",
                "--build", "GRCh38", "--annotation-source", "ensembl",
                "--output-dir", str(tmp_path),
            ],
            timeout=120,
        )
        assert r.returncode == 0
        assert "validate: ok" in r.combined

    def test_validate_unconfigured_build(self, tmp_path: Path) -> None:
        """§4.3 — T2T-CHM13 fails fast."""
        r = run_cli([
            "agentic-spliceai-ingest", "validate",
            "--build", "T2T-CHM13", "--annotation-source", "mane",
            "--output-dir", str(tmp_path),
        ])
        assert r.returncode == 1
        assert "validate: FAILED" in r.combined


# ---------------------------------------------------------------------------
# TestBase — agentic-spliceai-base
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cli("agentic-spliceai-base"),
                    reason="agentic-spliceai-base not installed")
class TestBase:
    """Smoke tests for ``agentic-spliceai-base``.

    Reference: ``dev/cli_reference/agentic-spliceai-base.md``.
    """

    def test_list_predictors_table(self) -> None:
        """§1.1 — three predictors registered."""
        r = run_cli(["agentic-spliceai-base", "list-predictors"])
        assert r.returncode == 0
        assert "spliceai" in r.combined
        assert "openspliceai" in r.combined
        assert "splicebert_classifier" in r.combined

    def test_list_predictors_json(self) -> None:
        """§1.2 — JSON shape parseable, has all 3."""
        import json

        r = run_cli(["agentic-spliceai-base", "list-predictors", "--format", "json"])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        names = {entry.get("name") for entry in data}
        assert {"spliceai", "openspliceai", "splicebert_classifier"}.issubset(names)

    @pytest.mark.bio
    @pytest.mark.slow
    def test_predict_openspliceai_tp53(self, tmp_path: Path) -> None:
        """§2.1 — OpenSpliceAI on TP53 produces ~19K positions."""
        r = run_cli(
            [
                "agentic-spliceai-base", "predict",
                "--predictor", "openspliceai",
                "--genes", "TP53",
                "--output-dir", str(tmp_path / "tp53"),
            ],
            timeout=120,
        )
        assert r.returncode == 0
        assert "PredictionResult(predictor=openspliceai" in r.combined
        assert "ok" in r.combined

    def test_predict_unknown_predictor(self, tmp_path: Path) -> None:
        """§2.5 — unknown predictor → exit 1, lists registered names."""
        r = run_cli([
            "agentic-spliceai-base", "predict",
            "--predictor", "unknown",
            "--genes", "BRCA1",
            "--output-dir", str(tmp_path / "x"),
        ])
        assert r.returncode == 1
        assert "Unknown predictor" in r.combined
        assert "spliceai" in r.combined  # in the registered-list message

    @pytest.mark.bio
    def test_predict_strict_preflight_aborts_on_partial(self, tmp_path: Path) -> None:
        """§2.6 — spliceai (GRCh37/ensembl) under --strict-preflight aborts."""
        r = run_cli([
            "agentic-spliceai-base", "predict",
            "--predictor", "spliceai",
            "--genes", "TP53",
            "--strict-preflight",
            "--output-dir", str(tmp_path / "x"),
        ])
        assert r.returncode == 2
        assert "Pre-flight" in r.combined
        assert "chromosome_split" in r.combined

    def test_predict_no_targets(self, tmp_path: Path) -> None:
        """§2.8 — predict without --genes or --chromosomes."""
        r = run_cli([
            "agentic-spliceai-base", "predict",
            "--predictor", "openspliceai",
        ])
        assert r.returncode == 2
        assert "required" in r.combined.lower()


# ---------------------------------------------------------------------------
# TestFeatures — agentic-spliceai-features
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cli("agentic-spliceai-features"),
                    reason="agentic-spliceai-features not installed")
class TestFeatures:
    """Smoke tests for ``agentic-spliceai-features``.

    Reference: ``dev/cli_reference/agentic-spliceai-features.md``.
    """

    # ------------------------- list-profiles -----------------------------

    def test_list_profiles_table(self) -> None:
        """§1.1 — all four profiles surface."""
        r = run_cli(["agentic-spliceai-features", "list-profiles"])
        assert r.returncode == 0
        for name in ("default", "full_stack", "isoform_discovery", "meta_m3_novel"):
            assert name in r.combined

    def test_list_profiles_json(self) -> None:
        """§1.2 — JSON list with the expected modality counts."""
        import json

        r = run_cli([
            "agentic-spliceai-features", "list-profiles", "--format", "json",
        ])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        names = {p["name"] for p in data}
        assert {"default", "full_stack"}.issubset(names)
        full_stack = next(p for p in data if p["name"] == "full_stack")
        assert full_stack["n_modalities"] == 9

    # ------------------------- list-tracks -------------------------------

    def test_list_tracks_grch38_full(self) -> None:
        """§2.1 — GRCh38 catalog has 12 entries (2 cons + 10 epigen).

        Note: the CLI's --format json emits a bare list (no {"tracks": ...}
        wrapper); the REST endpoint wraps. See "CLI/REST schema
        inconsistency" in dev/cli_reference/agentic-spliceai-features.md.
        """
        import json

        r = run_cli([
            "agentic-spliceai-features", "list-tracks",
            "--build", "GRCh38",
            "--format", "json",
        ])
        assert r.returncode == 0
        tracks = json.loads(r.stdout)
        assert len(tracks) == 12
        modalities = {t["modality"] for t in tracks}
        assert modalities == {"conservation", "epigenetic"}

    def test_list_tracks_conservation_only(self) -> None:
        """§2.2 — modality filter."""
        import json

        r = run_cli([
            "agentic-spliceai-features", "list-tracks",
            "--build", "GRCh38", "--modality", "conservation",
            "--format", "json",
        ])
        assert r.returncode == 0
        tracks = json.loads(r.stdout)
        assert len(tracks) == 2
        names = {t["name"] for t in tracks}
        assert names == {"phylop", "phastcons"}

    def test_list_tracks_grch37_no_epigenetic(self) -> None:
        """§2.4 — hg19 has conservation only (no ENCODE liftover here)."""
        import json

        r = run_cli([
            "agentic-spliceai-features", "list-tracks",
            "--build", "GRCh37",
            "--format", "json",
        ])
        assert r.returncode == 0
        tracks = json.loads(r.stdout)
        modalities = {t["modality"] for t in tracks}
        assert modalities == {"conservation"}

    # ------------------------- status ------------------------------------

    @pytest.mark.bio
    def test_status_canonical_mane_explicit_chroms(self) -> None:
        """§4.1 — production analysis_sequences/, four expected chroms ready."""
        r = run_cli([
            "agentic-spliceai-features", "status",
            "--canonical", "--build", "GRCh38", "--annotation-source", "mane",
            "--chromosomes", "1", "2", "21", "22",
        ])
        assert r.returncode == 0
        assert "Ready: True" in r.combined

    def test_status_empty_output_dir(self, tmp_path: Path) -> None:
        """§4.4 — empty output dir."""
        empty = tmp_path / "feat_empty"
        empty.mkdir()
        r = run_cli([
            "agentic-spliceai-features", "status",
            "--output-dir", str(empty),
        ])
        assert r.returncode == 1
        assert "Ready: False" in r.combined

    # ------------------------- prepare -----------------------------------

    def test_prepare_unknown_profile(self, tmp_path: Path) -> None:
        """§5.4 — unknown profile YAML."""
        r = run_cli([
            "agentic-spliceai-features", "prepare",
            "--profile", "does_not_exist",
            "--build", "GRCh38",
            "--input-dir", str(tmp_path / "preds"),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert r.returncode == 2
        assert "not found" in r.combined.lower()

    @pytest.mark.bio
    def test_prepare_dry_run(self, tmp_path: Path) -> None:
        """§5.1 — dry-run shows plan without writing."""
        r = run_cli([
            "agentic-spliceai-features", "prepare",
            "--profile", "default",
            "--build", "GRCh38",
            "--chromosomes", "22",
            "--input-dir", str(REPO_ROOT / "data" / "mane" / "GRCh38"
                               / "openspliceai_eval" / "precomputed"),
            "--output-dir", str(tmp_path / "feat_chr22"),
            "--dry-run",
        ])
        assert r.returncode == 0
        assert "Dry run" in r.combined
        assert "Planned steps" in r.combined


# ---------------------------------------------------------------------------
# TestIngestApi — REST endpoints under /api/ingest/*
# ---------------------------------------------------------------------------


def _server_up(host: str = "localhost", port: int = 8005, timeout: float = 1.0) -> bool:
    """Return True if the Bio Lab service responds on the given host:port."""
    try:
        import urllib.request

        urllib.request.urlopen(
            f"http://{host}:{port}/api/ingest/health",
            timeout=timeout,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _server_up(),
    reason="Bio Lab server not running on localhost:8005 — start with "
           "`conda run -n agentic-spliceai python -m server.bio.app`",
)
class TestIngestApi:
    """Smoke tests for the ingestion-layer REST endpoints.

    Reference: ``dev/cli_reference/api-ingest.md``.

    Tests skip when the server isn't running, so a developer can run
    ``pytest -m cli_smoke`` without first starting the service.
    """

    BASE_URL = "http://localhost:8005/api/ingest"

    def _get(self, path: str, **params) -> Tuple[int, dict]:
        """GET helper. Returns (status_code, parsed_json)."""
        import urllib.parse
        import urllib.request
        import json

        if params:
            qs = urllib.parse.urlencode(params)
            url = f"{self.BASE_URL}{path}?{qs}"
        else:
            url = f"{self.BASE_URL}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read())
            except Exception:
                payload = {}
            return exc.code, payload

    # ------------------------- health ------------------------------------

    def test_health(self) -> None:
        """§1.1"""
        code, body = self._get("/health")
        assert code == 200
        assert body.get("status") == "ok"

    # ------------------------- data-prep ---------------------------------

    def test_data_prep_builds(self) -> None:
        """§2.1 — registry-key field is ``id``; ``name`` is the display name."""
        code, body = self._get("/data-prep/builds")
        assert code == 200
        ids = {b.get("id") for b in body.get("builds", [])}
        assert {"spliceai", "openspliceai"}.issubset(ids)

    @pytest.mark.bio
    def test_data_prep_status_canonical_ready(self) -> None:
        """§3.1"""
        code, body = self._get(
            "/data-prep/status", build="GRCh38", annotation_source="mane",
        )
        assert code == 200
        assert body["ready"] is True
        assert set(body["artifacts"]) == {
            "gene_features", "splice_sites", "chromosome_split",
        }

    def test_data_prep_status_unconfigured_400(self) -> None:
        """§3.4"""
        code, body = self._get(
            "/data-prep/status",
            build="T2T-CHM13", annotation_source="mane",
        )
        assert code == 400
        assert "T2T-CHM13" in body.get("detail", "")

    def test_data_prep_status_no_params_400(self) -> None:
        """§3.6"""
        code, body = self._get("/data-prep/status")
        assert code == 400
        assert "Must pass" in body.get("detail", "")

    def test_data_prep_status_both_params_400(self) -> None:
        """§3.7"""
        code, body = self._get(
            "/data-prep/status",
            build="GRCh38", annotation_source="mane",
            output_dir="/tmp/x",
        )
        assert code == 400
        assert "either" in body.get("detail", "").lower()

    # ------------------------- features ----------------------------------

    def test_features_profiles(self) -> None:
        """§4.1"""
        code, body = self._get("/features/profiles")
        assert code == 200
        names = {p["name"] for p in body["profiles"]}
        assert {"default", "full_stack"}.issubset(names)

    def test_features_tracks_grch38_conservation(self) -> None:
        """§5.1"""
        code, body = self._get(
            "/features/tracks",
            build="GRCh38", modality="conservation",
        )
        assert code == 200
        names = {t["name"] for t in body["tracks"]}
        assert names == {"phylop", "phastcons"}

    @pytest.mark.bio
    def test_features_status_canonical(self) -> None:
        """§6.1"""
        code, body = self._get(
            "/features/status",
            build="GRCh38",
            annotation_source="mane",
            chromosomes="1,2,21,22",
        )
        assert code == 200
        assert body["ready"] is True
        assert set(body["chromosomes"]) == {"1", "2", "21", "22"}

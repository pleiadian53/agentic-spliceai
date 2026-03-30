"""
Foundation Model Embedding Diagnostics

Verifies that ``encode()`` returns per-nucleotide embeddings with the
correct shape and alignment — no stray CLS/EOS tokens, no off-by-one
errors, no silent padding.

These checks matter because downstream code (``_extract_single_strand``,
PCA fitting, scalar feature extraction) indexes embeddings by nucleotide
position.  A single extra row from an unstripped special token shifts
every downstream feature by one base pair.

Usage::

    # Quick smoke test (one model)
    python -m foundation_models.utils.embedding_diagnostics --model splicebert

    # All registered models
    python -m foundation_models.utils.embedding_diagnostics --model all

    # Verbose: print per-check details
    python -m foundation_models.utils.embedding_diagnostics --model hyenadna --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test sequences — varying lengths to catch edge cases
# ---------------------------------------------------------------------------

# Short (well within any model's context)
_SEQ_SHORT = "ATCGATCGATCG"  # 12 nt

# Medium (typical exon length)
_SEQ_MEDIUM = "ATCG" * 64  # 256 nt

# Near-boundary for SpliceBERT (max_context=510 or 1024)
_SEQ_NEAR_MAX = "ACGT" * 127  # 508 nt

# Sequences with edge-case nucleotides
_SEQ_WITH_N = "ATCGNNNNATCG"  # 12 nt, contains N

# Repeated single nucleotide (catches tokenizer quirks)
_SEQ_HOMO = "A" * 32  # 32 nt


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


@dataclass
class ModelDiagnostics:
    """Aggregated diagnostics for one model."""
    model_name: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_output_shape(
    model: "BaseEmbeddingModel",
    sequence: str,
    label: str,
) -> CheckResult:
    """Verify that encode() returns exactly (seq_len, hidden_dim).

    This is the most critical check: if the output has extra rows,
    downstream positional indexing will be misaligned.
    """
    seq_len = len(sequence)
    hidden_dim = model.metadata().hidden_dim

    emb = model.encode(sequence)
    if hasattr(emb, "cpu"):
        emb = emb.detach().cpu()

    actual_shape = tuple(emb.shape)
    expected_shape = (seq_len, hidden_dim)

    if actual_shape == expected_shape:
        return CheckResult(
            name=f"shape_{label}",
            passed=True,
            message=f"OK: {label} ({seq_len} nt) → {actual_shape}",
        )

    # Diagnose the mismatch
    row_diff = actual_shape[0] - seq_len
    if row_diff > 0:
        hint = (
            f"Output has {row_diff} extra row(s). "
            f"Likely unstripped special tokens (CLS/SEP/EOS). "
            f"Check whether encode() strips them."
        )
    elif row_diff < 0:
        hint = (
            f"Output is {-row_diff} row(s) short. "
            f"Possible truncation or missing padding."
        )
    else:
        hint = f"Row count matches but hidden_dim differs: got {actual_shape[1]}, expected {hidden_dim}."

    return CheckResult(
        name=f"shape_{label}",
        passed=False,
        message=f"FAIL: {label} ({seq_len} nt) → {actual_shape}, expected {expected_shape}",
        details=hint,
    )


def check_batch_consistency(
    model: "BaseEmbeddingModel",
) -> CheckResult:
    """Verify that single-sequence and batch encoding produce the same result.

    Catches bugs where batch padding or special-token handling differs
    between the single and batch code paths.
    """
    seq = _SEQ_SHORT

    emb_single = model.encode(seq)
    if hasattr(emb_single, "cpu"):
        emb_single = emb_single.detach().cpu().numpy()

    emb_batch = model.encode([seq])
    if hasattr(emb_batch, "cpu"):
        emb_batch = emb_batch.detach().cpu().numpy()

    # Batch output is [1, seq_len, hidden_dim]; squeeze to compare
    if emb_batch.ndim == 3:
        emb_batch = emb_batch[0]

    if emb_single.shape != emb_batch.shape:
        return CheckResult(
            name="batch_consistency",
            passed=False,
            message=(
                f"FAIL: single shape {emb_single.shape} != "
                f"batch shape {emb_batch.shape}"
            ),
        )

    max_diff = float(np.max(np.abs(emb_single.astype(np.float64) - emb_batch.astype(np.float64))))
    # Allow small floating-point differences (padding path may differ slightly)
    threshold = 1e-4
    passed = max_diff < threshold

    return CheckResult(
        name="batch_consistency",
        passed=passed,
        message=(
            f"{'OK' if passed else 'FAIL'}: "
            f"single vs batch max diff = {max_diff:.2e} "
            f"(threshold {threshold:.0e})"
        ),
        details=None if passed else "Single and batch paths produce different embeddings.",
    )


def check_determinism(
    model: "BaseEmbeddingModel",
) -> CheckResult:
    """Verify that two encode() calls on the same input produce identical output.

    Non-deterministic output (e.g., from dropout left enabled) would make
    PCA fitting and downstream features unreproducible.
    """
    seq = _SEQ_SHORT

    emb1 = model.encode(seq)
    emb2 = model.encode(seq)

    if hasattr(emb1, "cpu"):
        emb1 = emb1.detach().cpu().numpy()
    if hasattr(emb2, "cpu"):
        emb2 = emb2.detach().cpu().numpy()

    max_diff = float(np.max(np.abs(emb1.astype(np.float64) - emb2.astype(np.float64))))
    passed = max_diff == 0.0

    return CheckResult(
        name="determinism",
        passed=passed,
        message=(
            f"{'OK' if passed else 'WARN'}: "
            f"two calls max diff = {max_diff:.2e}"
        ),
        details=None if passed else (
            "Non-deterministic output. Check that model.eval() is called "
            "and dropout is disabled during inference."
        ),
    )


def check_no_nan_inf(
    model: "BaseEmbeddingModel",
) -> CheckResult:
    """Check that embeddings contain no NaN or Inf values."""
    seq = _SEQ_MEDIUM

    emb = model.encode(seq)
    if hasattr(emb, "cpu"):
        emb = emb.detach().cpu().numpy()

    has_nan = bool(np.any(np.isnan(emb)))
    has_inf = bool(np.any(np.isinf(emb)))

    if has_nan or has_inf:
        return CheckResult(
            name="no_nan_inf",
            passed=False,
            message=f"FAIL: NaN={has_nan}, Inf={has_inf}",
            details="Embeddings contain NaN or Inf. Check dtype and model precision.",
        )

    return CheckResult(
        name="no_nan_inf",
        passed=True,
        message="OK: no NaN or Inf in embeddings",
    )


def check_embedding_variation(
    model: "BaseEmbeddingModel",
) -> CheckResult:
    """Check that different input sequences produce different embeddings.

    Catches degenerate models that output constant embeddings (e.g.,
    broken weight loading).
    """
    emb_a = model.encode("ATCGATCG")
    emb_b = model.encode("GCTAGCTA")

    if hasattr(emb_a, "cpu"):
        emb_a = emb_a.detach().cpu().numpy()
    if hasattr(emb_b, "cpu"):
        emb_b = emb_b.detach().cpu().numpy()

    # Compare mean embeddings (sequences may differ in length)
    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)

    diff = float(np.linalg.norm(mean_a - mean_b))
    passed = diff > 1e-6

    return CheckResult(
        name="embedding_variation",
        passed=passed,
        message=f"{'OK' if passed else 'FAIL'}: ||mean_diff|| = {diff:.4e}",
        details=None if passed else "Different sequences produce identical embeddings.",
    )


def check_metadata_consistency(
    model: "BaseEmbeddingModel",
) -> CheckResult:
    """Check that metadata().hidden_dim matches actual embedding width."""
    meta = model.metadata()

    emb = model.encode(_SEQ_SHORT)
    if hasattr(emb, "cpu"):
        emb = emb.detach().cpu()

    actual_dim = emb.shape[-1]
    passed = actual_dim == meta.hidden_dim

    return CheckResult(
        name="metadata_consistency",
        passed=passed,
        message=(
            f"{'OK' if passed else 'FAIL'}: "
            f"metadata.hidden_dim={meta.hidden_dim}, actual={actual_dim}"
        ),
        details=None if passed else "metadata().hidden_dim does not match encode() output.",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_diagnostics(
    model: "BaseEmbeddingModel",
    verbose: bool = False,
) -> ModelDiagnostics:
    """Run all diagnostic checks on a loaded model.

    Parameters
    ----------
    model : BaseEmbeddingModel
        A loaded foundation model instance.
    verbose : bool
        Print per-check details.

    Returns
    -------
    ModelDiagnostics
        Aggregated results.
    """
    meta = model.metadata()
    diag = ModelDiagnostics(model_name=meta.name)

    # Shape checks across multiple sequence lengths
    test_sequences = [
        (_SEQ_SHORT, "short_12nt"),
        (_SEQ_MEDIUM, "medium_256nt"),
        (_SEQ_NEAR_MAX, "near_max_508nt"),
        (_SEQ_WITH_N, "with_N_12nt"),
        (_SEQ_HOMO, "homopolymer_32nt"),
    ]

    # Filter out sequences longer than model's max context
    max_ctx = meta.max_context
    test_sequences = [
        (seq, label) for seq, label in test_sequences
        if len(seq) <= max_ctx
    ]

    for seq, label in test_sequences:
        result = check_output_shape(model, seq, label)
        diag.checks.append(result)
        if verbose:
            _print_check(result)

    # Functional checks
    for check_fn in [
        check_batch_consistency,
        check_determinism,
        check_no_nan_inf,
        check_embedding_variation,
        check_metadata_consistency,
    ]:
        result = check_fn(model)
        diag.checks.append(result)
        if verbose:
            _print_check(result)

    return diag


def _print_check(result: CheckResult) -> None:
    """Print a single check result."""
    icon = "PASS" if result.passed else "FAIL"
    print(f"  [{icon}] {result.name}: {result.message}")
    if result.details:
        print(f"         {result.details}")


def print_report(diagnostics: ModelDiagnostics) -> None:
    """Print a human-readable report for one model."""
    meta_line = f"Model: {diagnostics.model_name}"
    print(f"\n{'=' * 60}")
    print(meta_line)
    print(f"{'=' * 60}")

    for result in diagnostics.checks:
        _print_check(result)

    n_total = len(diagnostics.checks)
    print(f"\n  Summary: {diagnostics.n_passed}/{n_total} passed", end="")
    if diagnostics.n_failed > 0:
        print(f", {diagnostics.n_failed} FAILED", end="")
    print()
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env file for HF_TOKEN and other secrets."""
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        # Walk up from this file to find .env at the repo root
        here = Path(__file__).resolve()
        for parent in here.parents:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logger.debug("Loaded %s", env_file)
                return
    except ImportError:
        pass


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(
        description="Verify foundation model embedding format and alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="all",
        help=(
            "Model name (e.g., splicebert, hyenadna, evo2) or 'all'. "
            "Default: all"
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-check details as they run",
    )
    args = parser.parse_args()

    from foundation_models.base import list_available_models, load_embedding_model

    if args.model.lower() == "all":
        model_names = list_available_models()
    else:
        model_names = [args.model]

    all_results: Dict[str, ModelDiagnostics] = {}
    any_failure = False

    for name in model_names:
        print(f"\nLoading {name}...")
        try:
            model = load_embedding_model(name)
        except Exception as e:
            print(f"  SKIP: could not load {name}: {e}")
            continue

        diag = run_diagnostics(model, verbose=args.verbose)
        print_report(diag)
        all_results[name] = diag

        if not diag.all_passed:
            any_failure = True

        # Free GPU memory between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 60}")
        for name, diag in all_results.items():
            status = "PASS" if diag.all_passed else "FAIL"
            print(f"  [{status}] {name}: {diag.n_passed}/{len(diag.checks)} checks passed")

    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()

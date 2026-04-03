#!/usr/bin/env python
"""Validate PCA artifacts from FM scalar extraction.

Checks whether the PCA components capture meaningful variation in the
foundation model embedding space. Run after Phase 1 of
07_streaming_fm_scalars.py produces pca_artifacts.npz.

Metrics:
  - Explained variance ratio (cumulative): how much of total variation
    is captured by the top-k components
  - Component norms: should be ~1.0 (unit vectors)
  - Component orthogonality: dot products should be ~0.0
  - Mean vector magnitude: sanity check on centering

Usage:
    python validate_fm_pca.py /path/to/pca_artifacts.npz

    # After scalar parquets are available, also validate features:
    python validate_fm_pca.py /path/to/pca_artifacts.npz \
        --scalars /path/to/fm_scalars_chr22.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def validate_pca(pca_path: Path) -> bool:
    """Validate PCA artifact structure and properties."""
    print(f"Loading PCA artifacts: {pca_path}")
    data = np.load(pca_path)

    keys = list(data.keys())
    print(f"  Keys: {keys}")

    required = {"components", "mean", "explained_variance_ratio"}
    missing = required - set(keys)
    if missing:
        print(f"  ERROR: missing keys: {missing}")
        return False

    components = data["components"]  # (k, hidden_dim)
    mean = data["mean"]              # (hidden_dim,)
    evr = data["explained_variance_ratio"]  # (k,)

    k, hidden_dim = components.shape
    print(f"\n  Components: {k} x {hidden_dim}")
    print(f"  Mean vector norm: {np.linalg.norm(mean):.4f}")

    # ── Explained variance ───────────────────────────────────────
    print(f"\n  Explained Variance Ratio:")
    cumulative = 0.0
    for i, v in enumerate(evr):
        cumulative += v
        print(f"    PC{i+1}: {v:.4f} (cumulative: {cumulative:.4f})")
    print(f"    Total (top-{k}): {cumulative:.4f}")

    if cumulative < 0.05:
        print(f"  WARNING: top-{k} components explain only {cumulative:.1%} of variance.")
        print(f"  This may indicate the embeddings have very high intrinsic dimensionality")
        print(f"  (expected for 4096-dim Evo2). PCA features may still be useful if they")
        print(f"  correlate with splice signal even without capturing majority variance.")
    elif cumulative > 0.5:
        print(f"  Good: top-{k} captures {cumulative:.1%} — strong low-rank structure.")

    # ── Component norms (should be ~1.0) ─────────────────────────
    norms = np.linalg.norm(components, axis=1)
    print(f"\n  Component norms: {norms}")
    if np.allclose(norms, 1.0, atol=0.01):
        print(f"  OK: all components are unit vectors")
    else:
        print(f"  WARNING: components are not unit-normalized (max deviation: {np.max(np.abs(norms - 1.0)):.4f})")

    # ── Orthogonality (dot products should be ~0.0) ──────────────
    gram = components @ components.T  # (k, k)
    off_diag = gram - np.diag(np.diag(gram))
    max_off_diag = np.max(np.abs(off_diag))
    print(f"\n  Orthogonality check:")
    print(f"    Max off-diagonal dot product: {max_off_diag:.6f}")
    if max_off_diag < 0.01:
        print(f"    OK: components are orthogonal")
    else:
        print(f"    WARNING: non-trivial correlation between components")

    # ── Optional: centroid vectors ───────────────────────────────
    if "donor_centroid" in data and "acceptor_centroid" in data:
        dc = data["donor_centroid"]
        ac = data["acceptor_centroid"]
        cos_sim = np.dot(dc, ac) / (np.linalg.norm(dc) * np.linalg.norm(ac))
        print(f"\n  Centroid vectors present:")
        print(f"    Donor norm: {np.linalg.norm(dc):.4f}")
        print(f"    Acceptor norm: {np.linalg.norm(ac):.4f}")
        print(f"    Cosine similarity (donor vs acceptor): {cos_sim:.4f}")
        if abs(cos_sim) > 0.95:
            print(f"    WARNING: centroids nearly identical — limited discriminative value")

    return True


def validate_scalars(scalar_path: Path, pca_path: Path) -> None:
    """Validate scalar feature distributions from a chromosome parquet."""
    import polars as pl

    print(f"\nLoading scalars: {scalar_path}")
    df = pl.read_parquet(scalar_path)
    n = df.height
    print(f"  Positions: {n:,}")
    print(f"  Columns: {df.columns}")

    pca_cols = [c for c in df.columns if c.startswith("fm_pca_")]
    print(f"\n  Feature distributions:")
    print(f"  {'Column':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Nulls':>8}")
    print(f"  {'-'*70}")

    for col in sorted(c for c in df.columns if c.startswith("fm_")):
        series = df[col]
        nulls = series.null_count()
        valid = series.drop_nulls()
        if valid.len() == 0:
            print(f"  {col:<22} {'ALL NULL':>10}")
            continue
        arr = valid.to_numpy()
        # Use nan-safe stats: gradient values are stored as float NaN
        # (from Python's float('nan') at gene boundaries), not Polars
        # nulls — so drop_nulls() doesn't filter them.
        n_nan = int(np.isnan(arr).sum())
        print(f"  {col:<22} {np.nanmean(arr):>10.4f} {np.nanstd(arr):>10.4f} "
              f"{np.nanmin(arr):>10.4f} {np.nanmax(arr):>10.4f} {nulls + n_nan:>8}")

    # ── PCA component correlations ───────────────────────────────
    if len(pca_cols) >= 2:
        print(f"\n  PCA component correlations (should be ~0 if PCA is correct):")
        pca_df = df.select(pca_cols).drop_nulls()
        if pca_df.height > 100:
            pca_arr = pca_df.to_numpy()
            corr = np.corrcoef(pca_arr.T)
            for i in range(len(pca_cols)):
                for j in range(i + 1, len(pca_cols)):
                    r = corr[i, j]
                    flag = " ***" if abs(r) > 0.1 else ""
                    print(f"    {pca_cols[i]} vs {pca_cols[j]}: r={r:.4f}{flag}")

    # ── Gradient distribution ────────────────────────────────────
    if "fm_local_gradient" in df.columns:
        raw = df["fm_local_gradient"].to_numpy()
        # Gradient uses float NaN for gene boundaries (not Polars null),
        # so filter with np.isnan rather than drop_nulls().
        n_nan = int(np.isnan(raw).sum())
        grad = raw[~np.isnan(raw)]
        print(f"\n  Gradient analysis:")
        print(f"    Valid: {len(grad):,} ({len(grad)/n*100:.1f}%)")
        print(f"    NaN (gene boundaries): {n_nan:,} ({n_nan/n*100:.1f}%)")
        if len(grad) > 0:
            pcts = np.percentile(grad, [25, 50, 75, 90, 99])
            print(f"    Percentiles [25/50/75/90/99]: "
                  f"{pcts[0]:.2f} / {pcts[1]:.2f} / {pcts[2]:.2f} / "
                  f"{pcts[3]:.2f} / {pcts[4]:.2f}")

    # ── Embedding norm distribution ──────────────────────────────
    if "fm_embedding_norm" in df.columns:
        norms = df["fm_embedding_norm"].drop_nulls().to_numpy()
        print(f"\n  Embedding norm analysis:")
        print(f"    Mean: {norms.mean():.2f}, Std: {norms.std():.2f}")
        print(f"    Range: [{norms.min():.2f}, {norms.max():.2f}]")
        cv = norms.std() / norms.mean() if norms.mean() > 0 else 0
        print(f"    CV (coefficient of variation): {cv:.4f}")
        if cv < 0.01:
            print(f"    WARNING: very low variation — norms are nearly constant")
            print(f"    This may mean the model produces uniform-magnitude embeddings")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate PCA artifacts and FM scalar features",
    )
    parser.add_argument(
        "pca_path", type=Path,
        help="Path to pca_artifacts.npz",
    )
    parser.add_argument(
        "--scalars", type=Path, default=None,
        help="Optional: path to fm_scalars_{chrom}.parquet for feature validation",
    )
    args = parser.parse_args()

    if not args.pca_path.exists():
        print(f"Error: {args.pca_path} not found")
        return 1

    print("=" * 60)
    print("FM PCA Validation")
    print("=" * 60)

    ok = validate_pca(args.pca_path)

    if args.scalars is not None:
        if not args.scalars.exists():
            print(f"\nError: {args.scalars} not found")
            return 1
        validate_scalars(args.scalars, args.pca_path)

    print("\n" + "=" * 60)
    print("PASSED" if ok else "FAILED")
    print("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

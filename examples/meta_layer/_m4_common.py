"""Shared helpers for the M4 regulator-arm benchmark (scripts 17+).

M4 task (HepG2 RBP knockdown, A3SS/A5SS): given a local alternative-splicing event,
a knocked-down regulator R, and features available *before* the knockdown outcome is
observed, predict the signed change in relative usage of the event's competing
splice sites, ``dpsi`` = ΔPSI of the long form.

This module owns the parts every M4 script must agree on:

* **Labels + bookkeeping** (:func:`load_events`): the prespecified significant set,
  event-level weights, and a leave-this-experiment-out baseline PSI.
* **Features**: per-geometry site features of the long, short, and partner sites
  (:func:`geometry_features`) and per-(event, R) binding of R
  (:func:`regulator_features`). Kept separate because site features do not depend on
  R: 19K geometries, not 1.6M rows.
* **Splits** (:func:`assign_gene_split`, :func:`regulator_folds`): gene-grouped,
  because the long/short forms are algebraically coupled and the same event recurs
  under every regulator with a different per-file rMATS ID.
* **Metrics** (:func:`regression_metrics`): magnitude and direction, reported together.

Inputs (built by ``examples/data_preparation/m4/``): ``kd_events.parquet`` (01),
``site_features.parquet`` (03), ``regulator_site_features.parquet`` (04).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
M4_LABELS = REPO / "data/mane/GRCh38/m4_labels"
EVENTS = M4_LABELS / "kd_events.parquet"
SITE_FEATURES = M4_LABELS / "site_features.parquet"
REGULATOR_FEATURES = M4_LABELS / "regulator_site_features.parquet"

KEY = ["chrom", "position", "strand"]
ROLES = ("long", "short", "partner")

# Prespecified "significant, substantial" events: the direction-accuracy set.
SIG_FDR = 0.05
SIG_ABS_DPSI = 0.1
SIG_MIN_READS = 20  # per condition (inclusion + skipping junction reads, summed over replicates)

# Columns that are, or are computed from, the knockdown outcome. Never model inputs.
OUTCOME_COLS = frozenset({
    "dpsi", "psi_kd", "fdr", "pvalue", "ijc_kd", "sjc_kd", "n_rep_kd",
    # the KD's own control shares sampling noise with dpsi (regression to the mean);
    # use baseline_psi_loo instead
    "psi_ctrl", "ijc_ctrl", "sjc_ctrl", "n_rep_ctrl",
})

REGULATOR_COLS = ("bind_w50", "bind_w300", "bind_w1000", "bind_intron300", "bind_exon300", "dist")


# ── Labels ───────────────────────────────────────────────────────────────


def load_events(path: Path = EVENTS) -> pl.DataFrame:
    """Event × regulator rows with label bookkeeping.

    Adds ``reads_kd`` / ``reads_ctrl``, the prespecified ``significant`` flag,
    ``weight`` (each (regulator, alternative-site pair) cluster sums to 1, so a pair
    measured against several flanking exons is not over-counted), and the pre-outcome
    baseline columns ``baseline_psi_loo`` and ``n_other_kd``.

    ``baseline_psi_loo`` leaves out the row's whole **control batch**, not just the row:
    knockdowns in one batch share control libraries, so their control PSIs carry the same
    sampling noise as this row's ΔPSI (corr with ΔPSI: own control −0.247, row-out
    −0.047, batch-out −0.025).
    """
    ev = pl.read_parquet(path).filter(pl.col("dpsi").is_not_null())
    ev = ev.with_columns(
        (pl.col("ijc_kd") + pl.col("sjc_kd")).alias("reads_kd"),
        (pl.col("ijc_ctrl") + pl.col("sjc_ctrl")).alias("reads_ctrl"),
    )
    ev = ev.with_columns(
        (
            (pl.col("fdr") < SIG_FDR)
            & (pl.col("dpsi").abs() >= SIG_ABS_DPSI)
            & (pl.col("reads_kd") >= SIG_MIN_READS)
            & (pl.col("reads_ctrl") >= SIG_MIN_READS)
        ).alias("significant"),
        (1.0 / pl.len().over(["rbp", "altpair_key"])).alias("weight"),
    )
    # Leave-this-control-batch-out baseline: the geometry's mean control PSI over the
    # control samples of OTHER batches. Pre-outcome, and independent of this row's
    # control noise.
    psi = pl.col("psi_ctrl")
    n_out = psi.count().over("event_key") - psi.count().over(["event_key", "control_batch"])
    s_out = psi.sum().over("event_key") - psi.sum().over(["event_key", "control_batch"])
    return ev.with_columns(
        pl.when(n_out > 0).then(s_out / n_out).otherwise(None).alias("baseline_psi_loo"),
        (pl.len().over("event_key") - 1).alias("n_other_kd"),
    )


# ── Features ─────────────────────────────────────────────────────────────


def site_feature_columns(site_features: pl.DataFrame) -> list[str]:
    """The meta-layer model-input columns present in the site table (MODALITY_COLUMNS)."""
    from agentic_spliceai.splice_engine.meta_layer.training.data_utils import MODALITY_COLUMNS

    present = set(site_features.columns)
    return [c for cols in MODALITY_COLUMNS.values() for c in cols if c in present]


def geometry_features(events: pl.DataFrame, site_features: pl.DataFrame,
                      deltas: bool = True) -> tuple[pl.DataFrame, list[str]]:
    """One row per ``event_key``: event geometry + the three sites' feature vectors.

    Columns are prefixed ``long__``, ``short__``, ``partner__``; with ``deltas``, also
    ``delta__<c>`` = long − short (the competition the label measures). Geometry is
    strand-free: alternative-site distance, the two junction spans, exon lengths.
    """
    cols = site_feature_columns(site_features)
    sf = site_features.select(*KEY, *cols, "gene_set").unique(KEY)
    geo = events.select(
        "event_key", "chrom", "strand", "event_type",
        "long_site", "short_site", "partner_site",
        "long_exon_start", "long_exon_end", "short_exon_start", "short_exon_end",
        "flank_exon_start", "flank_exon_end",
    ).unique("event_key")
    geo = geo.with_columns(
        (pl.col("event_type") == "A5SS").cast(pl.Int8).alias("is_a5ss"),
        (pl.col("long_site") - pl.col("short_site")).abs().alias("alt_dist"),
        (pl.col("partner_site") - pl.col("long_site")).abs().alias("junction_span_long"),
        (pl.col("partner_site") - pl.col("short_site")).abs().alias("junction_span_short"),
        (pl.col("long_exon_end") - pl.col("long_exon_start") + 1).alias("long_exon_len"),
        (pl.col("short_exon_end") - pl.col("short_exon_start") + 1).alias("short_exon_len"),
        (pl.col("flank_exon_end") - pl.col("flank_exon_start") + 1).alias("flank_exon_len"),
    )
    geo_cols = ["is_a5ss", "alt_dist", "junction_span_long", "junction_span_short",
                "long_exon_len", "short_exon_len", "flank_exon_len"]
    out = geo.select("event_key", "chrom", "strand", *[f"{r}_site" for r in ROLES], *geo_cols)
    for role in ROLES:
        renamed = sf.rename({c: f"{role}__{c}" for c in cols}).rename(
            {"position": f"{role}_site", "gene_set": f"{role}__gene_set"})
        out = out.join(renamed, on=["chrom", f"{role}_site", "strand"], how="left")
    feat_cols = geo_cols + [f"{r}__{c}" for r in ROLES for c in cols]
    if deltas:
        out = out.with_columns(
            [(pl.col(f"long__{c}") - pl.col(f"short__{c}")).alias(f"delta__{c}") for c in cols])
        feat_cols += [f"delta__{c}" for c in cols]
    out = out.with_columns(
        pl.when(pl.col("long__gene_set") == "mane").then(pl.lit("mane"))
        .otherwise(pl.col("long__gene_set")).alias("gene_set"))
    return out.drop("chrom", "strand", *[f"{r}_site" for r in ROLES],
                    *[f"{r}__gene_set" for r in ROLES]), feat_cols


def regulator_features(events: pl.DataFrame,
                       regulator_site_features: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """One row per (event_key, rbp): R's binding near the long, short, and partner sites.

    Null where R has no eCLIP (``eclip_cell == "none"``); ``has_eclip`` and
    ``eclip_hepg2`` say which, so missingness is explicit rather than imputed.
    """
    rf = regulator_site_features.select(*KEY, "rbp", "eclip_cell", *REGULATOR_COLS)
    out = events.select("event_key", "rbp", "chrom", "strand",
                        *[f"{r}_site" for r in ROLES])
    for role in ROLES:
        renamed = rf.rename({c: f"R_{role}__{c}" for c in REGULATOR_COLS}).rename(
            {"position": f"{role}_site", "eclip_cell": f"_cell_{role}"})
        out = out.join(renamed, on=["chrom", f"{role}_site", "strand", "rbp"], how="left")
    out = out.with_columns(
        (pl.col("_cell_long") != "none").cast(pl.Int8).alias("has_eclip"),
        (pl.col("_cell_long") == "HepG2").cast(pl.Int8).alias("eclip_hepg2"),
    )
    cols = ["has_eclip", "eclip_hepg2"] + [f"R_{r}__{c}" for r in ROLES for c in REGULATOR_COLS]
    return out.select("event_key", "rbp", *cols), cols


# ── Splits ───────────────────────────────────────────────────────────────


def assign_gene_split(events: pl.DataFrame, preset: str = "spliceai",
                      seed: int = 42) -> pl.DataFrame:
    """Add ``split`` ∈ {train, val, test} by gene (SpliceAI preset: test = chr1/3/5/7/9).

    Gene grouping keeps every representation of an event (long/short forms, every
    regulator, every flanking exon of an alternative-site pair) in one split.
    """
    from agentic_spliceai.splice_engine.meta_layer.training.data_utils import get_gene_split

    gs = get_gene_split(events.select("gene_id", "chrom").unique(), preset=preset, seed=seed)
    lookup = pl.DataFrame({
        "gene_id": [*gs.train_genes, *gs.val_genes, *gs.test_genes],
        "split": (["train"] * len(gs.train_genes) + ["val"] * len(gs.val_genes)
                  + ["test"] * len(gs.test_genes)),
    })
    out = events.join(lookup, on="gene_id", how="left")
    n_missing = out["split"].null_count()
    if n_missing:
        raise ValueError(f"{n_missing} event rows have no gene split assignment")
    return out


def regulator_folds(rbps: list[str], n_folds: int = 5, seed: int = 0,
                    groups: dict[str, str] | None = None) -> dict[str, int]:
    """Assign each regulator to one of ``n_folds`` held-out folds (fixed seed).

    With ``groups`` (regulator → control batch), whole groups go to one fold, largest
    first onto the currently smallest fold. Regulators sharing control samples share
    control noise in their ΔPSI, so splitting a batch across folds leaks the held-out
    regulator's noise into training (it inflated the S2 event prior from 0.655 to
    0.742 direction accuracy).
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(sorted(rbps))
        return {r: i % n_folds for i, r in enumerate(order)}
    members: dict[str, list[str]] = {}
    for r in sorted(rbps):
        members.setdefault(groups[r], []).append(r)
    keys = list(rng.permutation(sorted(members)))
    keys.sort(key=lambda g: -len(members[g]))  # stable: ties keep the seeded order
    load = [0] * n_folds
    out: dict[str, int] = {}
    for g in keys:
        f = int(np.argmin(load))
        load[f] += len(members[g])
        out.update(dict.fromkeys(members[g], f))
    return out


# ── Metrics ──────────────────────────────────────────────────────────────


def regression_metrics(y: np.ndarray, yhat: np.ndarray, weight: np.ndarray,
                       significant: np.ndarray) -> dict[str, float]:
    """Magnitude and direction for one regulator (or any subset).

    * all events: Spearman ρ, Pearson r, weighted MAE.
    * significant events: direction accuracy (sign agreement), sign AUROC, MAE.
    Direction is only scored where it is defined by the prespecified significant set,
    and it is always reported next to magnitude error.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    out: dict[str, float] = {"n": float(len(y)), "n_sig": float(significant.sum())}
    if len(y) >= 3 and np.std(yhat) > 0 and np.std(y) > 0:
        out["spearman"] = float(spearmanr(y, yhat).statistic)
        out["pearson"] = float(pearsonr(y, yhat).statistic)
    out["mae_w"] = float(np.average(np.abs(y - yhat), weights=weight))
    ys, ps = y[significant], yhat[significant]
    if len(ys):
        # a prediction of exactly 0 has no direction: score it as a coin flip (0.5)
        agree = np.where(ps == 0, 0.5, (np.sign(ps) == np.sign(ys)).astype(float))
        out["dir_acc_sig"] = float(np.mean(agree))
        out["mae_sig"] = float(np.mean(np.abs(ys - ps)))
        if 0 < (ys > 0).sum() < len(ys):
            out["sign_auroc_sig"] = float(roc_auc_score(ys > 0, ps))
    return out


def macro(per_regulator: pl.DataFrame, cols: list[str]) -> dict[str, float]:
    """Unweighted mean over regulators (each regulator is one task)."""
    return {c: float(per_regulator[c].mean()) for c in cols if c in per_regulator.columns}

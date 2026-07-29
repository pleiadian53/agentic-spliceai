"""Phase 1 (characterization) — summarize the M4 ΔPSI corpus + reconcile with eCLIP.

Reads the label corpus built by `01_ingest_kd_dpsi.py` and the eCLIP binding panel,
then writes a human-readable report + a machine-readable stats blob. These numbers
are what the *next* step (M4 training formulation) is planned against: the ΔPSI
distribution and event-level sign balance decide the label thresholds and loss;
the KD∩eCLIP regulator intersection is the set for which both a perturbation effect
and binding evidence exist — the natural starting panel for regulator conditioning.

Run (after 01):
    python examples/data_preparation/m4/02_characterize_corpus.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "data/mane/GRCh38/m4_labels/kd_dpsi.parquet"
ECLIP = REPO / "data/mane/GRCh38/rbp_data/eclip_peaks_neuronal.parquet"
OUT_DIR = REPO / "output/meta_layer/m4_labels"

# Event-level significance thresholds reported (not applied to the corpus).
SIG = {"fdr05": 0.05, "fdr01": 0.01, "dpsi": 0.1}
DPSI_BINS = [-1.01, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.01]


def _quantiles(s: pl.Series) -> dict[str, float]:
    """Summary quantiles of a numeric series (nulls ignored by polars)."""
    return {
        "mean": float(s.mean()),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
    }


def _histogram(s: pl.Series, edges: list[float]) -> list[tuple[str, int]]:
    """Count values of `s` in the half-open bins defined by `edges`."""
    out: list[tuple[str, int]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        n = int(s.filter((s >= lo) & (s < hi)).len())
        out.append((f"[{lo:+.2f}, {hi:+.2f})", n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Characterize the M4 ΔPSI label corpus.")
    ap.add_argument(
        "--top-n", type=int, default=25, help="top regulators to list by significant events"
    )
    args = ap.parse_args()

    df = pl.read_parquet(CORPUS)
    events = df.filter(pl.col("form") == "long")  # one row per (event, RBP)

    stats: dict = {}

    # --- Counts ---------------------------------------------------------------
    stats["n_site_rows"] = df.height
    stats["n_events"] = events.height
    stats["n_unique_sites"] = df.select(["chrom", "position", "strand", "splice_type"]).n_unique()
    stats["n_rbps"] = int(df["rbp"].n_unique())
    stats["events_by_event_type"] = {
        r["event_type"]: r["len"]
        for r in events.group_by("event_type").len().sort("event_type").iter_rows(named=True)
    }
    stats["rows_by_splice_type"] = {
        r["splice_type"]: r["len"]
        for r in df.group_by("splice_type").len().sort("splice_type").iter_rows(named=True)
    }

    # --- Signed ΔPSI distribution per form ------------------------------------
    stats["dpsi_by_form"] = {
        form: _quantiles(df.filter(pl.col("form") == form)["dpsi"].drop_nulls())
        for form in ("long", "short")
    }
    stats["dpsi_long_histogram"] = _histogram(
        df.filter(pl.col("form") == "long")["dpsi"].drop_nulls(), DPSI_BINS
    )

    # --- Sign balance ---------------------------------------------------------
    site_pos = int((df["dpsi"] > 0).sum())
    site_neg = int((df["dpsi"] < 0).sum())
    site_zero = int((df["dpsi"] == 0).sum())
    stats["sign_balance_site_level"] = {
        "pos": site_pos,
        "neg": site_neg,
        "zero": site_zero,
        "note": "structurally ~50/50: each event emits +dpsi and -dpsi",
    }
    sig_events = events.filter(
        (pl.col("fdr") < SIG["fdr05"]) & (pl.col("inclvl_diff").abs() >= SIG["dpsi"])
    )
    ev_pos = int((sig_events["inclvl_diff"] > 0).sum())
    ev_neg = int((sig_events["inclvl_diff"] < 0).sum())
    stats["sign_balance_event_level_significant"] = {
        "pos_derepressed": ev_pos,
        "neg_repressed": ev_neg,
        "n": ev_pos + ev_neg,
        "note": "sign of IncLevelDifference among FDR<0.05 & |dPSI|>=0.1 events (the meaningful grain)",
    }

    # --- Significance fractions (event level) ---------------------------------
    stats["significant_events"] = {
        "fdr<0.05": int((events["fdr"] < SIG["fdr05"]).sum()),
        "fdr<0.01": int((events["fdr"] < SIG["fdr01"]).sum()),
        "|dpsi|>=0.1": int((events["inclvl_diff"].abs() >= SIG["dpsi"]).sum()),
        "fdr<0.05 & |dpsi|>=0.1": sig_events.height,
    }

    # --- Canonical GT/AG rate by strand x splice_type x form ------------------
    canon = (
        df.group_by(["strand", "splice_type", "form"])
        .agg(pl.col("canonical_dinuc").mean().alias("rate"), pl.len().alias("n"))
        .sort(["strand", "splice_type", "form"])
    )
    stats["canonical_rate_overall"] = float(df["canonical_dinuc"].mean())
    stats["canonical_rate_by_group"] = [
        {
            "strand": r["strand"],
            "splice_type": r["splice_type"],
            "form": r["form"],
            "rate": round(r["rate"], 4),
            "n": r["n"],
        }
        for r in canon.iter_rows(named=True)
    ]

    # --- Per-RBP significant-event distribution -------------------------------
    by_rbp = sig_events.group_by("rbp").len().sort("len", descending=True)
    stats["per_rbp_significant"] = {
        "top": [
            {"rbp": r["rbp"], "n": r["len"]} for r in by_rbp.head(args.top_n).iter_rows(named=True)
        ],
        "min": int(by_rbp["len"].min()) if by_rbp.height else 0,
        "median": float(by_rbp["len"].median()) if by_rbp.height else 0.0,
        "max": int(by_rbp["len"].max()) if by_rbp.height else 0,
    }
    all_rbps = set(df["rbp"].unique().to_list())
    rbps_with_sig = set(by_rbp["rbp"].to_list())
    stats["rbps_with_zero_significant_events"] = sorted(all_rbps - rbps_with_sig)

    # --- KD ∩ eCLIP reconciliation --------------------------------------------
    eclip_rbps = set(pl.read_parquet(ECLIP, columns=["rbp"])["rbp"].unique().to_list())
    kd_rbps = all_rbps
    both = sorted(kd_rbps & eclip_rbps)
    stats["rbp_panels"] = {
        "n_kd": len(kd_rbps),
        "n_eclip": len(eclip_rbps),
        "n_both": len(both),
        "n_kd_only": len(kd_rbps - eclip_rbps),
        "n_eclip_only": len(eclip_rbps - kd_rbps),
        "both": both,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    _write_report(OUT_DIR / "corpus_report.md", stats)
    print(f"Wrote {OUT_DIR / 'corpus_stats.json'}")
    print(f"Wrote {OUT_DIR / 'corpus_report.md'}")
    print(
        f"\n{stats['n_site_rows']:,} rows · {stats['n_events']:,} events · {stats['n_rbps']} RBPs · "
        f"{stats['n_unique_sites']:,} unique sites · canonical {stats['canonical_rate_overall']:.4f}"
    )
    print(
        f"significant events (FDR<0.05 & |dPSI|>=0.1): {sig_events.height:,} "
        f"({ev_pos:,} de-repressed / {ev_neg:,} repressed)"
    )
    print(f"regulators with both KD effect + eCLIP binding: {len(both)}")


def _write_report(path: Path, s: dict) -> None:
    """Render the stats blob as a readable markdown report."""
    L: list[str] = []
    L.append("# M4 perturbation-paired ΔPSI corpus — characterization\n")
    L.append("Source: ENCODE shRNA-knockdown rMATS JCEC (SpliceTools, K562/HepG2), A3SS/A5SS.")
    L.append(
        "Built by `examples/data_preparation/m4/01_ingest_kd_dpsi.py`; "
        "characterized by `02_characterize_corpus.py`.\n"
    )

    L.append("## Corpus size")
    L.append(
        f"- **{s['n_site_rows']:,}** signed per-site rows "
        f"(= 2 × {s['n_events']:,} events, Option B long+short forms)"
    )
    L.append(
        f"- **{s['n_rbps']}** regulators (RBPs) · **{s['n_unique_sites']:,}** unique regulated splice sites"
    )
    L.append(
        "- events by type: " + ", ".join(f"{k} {v:,}" for k, v in s["events_by_event_type"].items())
    )
    L.append(
        "- rows by splice type: "
        + ", ".join(f"{k} {v:,}" for k, v in s["rows_by_splice_type"].items())
        + "\n"
    )

    L.append("## Coordinate validation (GT/AG oracle, by strand × splice_type × form)")
    L.append(
        f"Overall canonical rate **{s['canonical_rate_overall']:.4f}** (hard-fail gate < 0.85).\n"
    )
    L.append("| strand | splice_type | form | rate | n |")
    L.append("|---|---|---|---:|---:|")
    for r in s["canonical_rate_by_group"]:
        L.append(
            f"| {r['strand']} | {r['splice_type']} | {r['form']} | {r['rate']:.4f} | {r['n']:,} |"
        )
    L.append("")

    L.append("## Signed ΔPSI distribution")
    L.append("| form | mean | q05 | q25 | median | q75 | q95 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for form, q in s["dpsi_by_form"].items():
        L.append(
            f"| {form} | {q['mean']:+.3f} | {q['q05']:+.3f} | {q['q25']:+.3f} | "
            f"{q['median']:+.3f} | {q['q75']:+.3f} | {q['q95']:+.3f} |"
        )
    L.append("\nLong-form ΔPSI histogram:")
    L.append("| bin | n |")
    L.append("|---|---:|")
    for label, n in s["dpsi_long_histogram"]:
        L.append(f"| {label} | {n:,} |")
    L.append("")

    L.append("## Sign balance")
    sb = s["sign_balance_site_level"]
    L.append(
        f"- **Site level** (structural): +{sb['pos']:,} / −{sb['neg']:,} / 0={sb['zero']:,} "
        f"— {sb['note']}."
    )
    eb = s["sign_balance_event_level_significant"]
    L.append(
        f"- **Event level, significant** (the meaningful grain): "
        f"**{eb['pos_derepressed']:,} de-repressed (+) / {eb['neg_repressed']:,} repressed (−)** "
        f"of {eb['n']:,} — both directions present.\n"
    )

    L.append("## Significance (event level, reported not filtered)")
    for k, v in s["significant_events"].items():
        L.append(f"- {k}: **{v:,}** events")
    L.append("")

    L.append("## Regulators")
    pr = s["per_rbp_significant"]
    L.append(
        f"Significant events per RBP — min {pr['min']}, median {pr['median']:.0f}, max {pr['max']}. "
        f"Top {len(pr['top'])} regulators (spliceosome/SR/EJC core = a biology sanity check):\n"
    )
    L.append("| rank | RBP | significant events |")
    L.append("|---:|---|---:|")
    for i, r in enumerate(pr["top"], 1):
        L.append(f"| {i} | {r['rbp']} | {r['n']:,} |")
    zero = s["rbps_with_zero_significant_events"]
    L.append(
        f"\nRBPs with 0 significant events (FDR<0.05 & |dPSI|≥0.1): "
        f"{len(zero)}" + (f" — {', '.join(zero)}" if zero else "") + "\n"
    )

    L.append("## KD ∩ eCLIP regulator reconciliation")
    p = s["rbp_panels"]
    L.append(
        f"- KD-effect panel: **{p['n_kd']}** RBPs · eCLIP-binding panel "
        f"(`eclip_peaks_neuronal.parquet`): **{p['n_eclip']}** RBPs"
    )
    L.append(
        f"- **{p['n_both']}** regulators have both a KD effect table and eCLIP binding evidence "
        f"({p['n_kd_only']} KD-only, {p['n_eclip_only']} eCLIP-only)"
    )
    L.append(
        f"- This {p['n_both']}-RBP intersection is the natural starting panel for a "
        f"regulator-conditioned M4.\n"
    )
    L.append(
        "<details><summary>The "
        + str(p["n_both"])
        + " regulators with both effect + binding</summary>\n"
    )
    L.append(", ".join(p["both"]))
    L.append("\n</details>")

    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

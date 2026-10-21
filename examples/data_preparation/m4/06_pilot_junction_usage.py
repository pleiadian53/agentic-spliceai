#!/usr/bin/env python
"""Label QC for knockdown junction counts: are position-level labels usable?

Run after ``05_extract_kd_junctions.py`` and before building position-level labels. Reads its
per-BAM junction tables and answers five questions for each knockdown:

1. **Quality.** Reads, spliced reads, junctions, and the share of junction reads with a canonical
   splice motif (from ``manifest.json``).
2. **Site usage and its change.** For every splice site, its *usage* is its share of the reads
   from its partner sites. For an acceptor ``a``, the partners are the donors joined to it, and

       usage(a) = reads into a / reads out of those donors to any acceptor

   (donors mirror this). Usage is computed separately in knockdown and control, so a site unused
   in control and used after knockdown gets usage 0 → x.
3. **Agreement with rMATS.** For the knockdown's A3SS/A5SS events in ``kd_events.parquet``, the
   long-form ΔPSI recomputed from junction reads alone is compared with rMATS's ΔPSI (which also
   uses exon-body reads).
4. **Induced sites.** Sites with control usage ≤ 0.01 and knockdown usage ≥ 0.10 (each with at
   least ``--min-den`` partner reads, and at least 5 reads into the site after knockdown), split
   by whether MANE or Ensembl annotate them.
5. **Noise floor.** How many "induced" sites appear when nothing was knocked down: one control
   replicate is compared with the other replicate of the same control experiment (both ways).
   Each knockdown replicate is then compared with a single control replicate, so both sides have
   one replicate's depth, and the ratio of the two counts is the knockdown's excess over noise.
   An excess says the knockdown induces more sites than chance, not that any one site is real.

Replicate agreement (ΔPSI from replicate 1 vs replicate 2, each against pooled control) is
reported as a measure of label noise.

Output: ``<junction-dir>/pilot_report.json`` and a printed summary. A few knockdowns run in
seconds on a laptop (junction tables are a few MB each); all of them belong on the pod.

Usage:
    python examples/data_preparation/m4/06_pilot_junction_usage.py
    python examples/data_preparation/m4/06_pilot_junction_usage.py --targets U2AF1 PTBP1
    python examples/data_preparation/m4/06_pilot_junction_usage.py --workers 12   # pod, all targets
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
JUNC = REPO / "data/mane/GRCh38/m4_junctions"
EVENTS = REPO / "data/mane/GRCh38/m4_labels/kd_events.parquet"
ANNOT = {"mane": REPO / "data/mane/GRCh38/splice_sites_enhanced.tsv",
         "ensembl": REPO / "data/ensembl/GRCh38/splice_sites_enhanced.tsv"}
GENES = REPO / "data/mane/GRCh38/gene_features.tsv"
J = ["chrom", "strand", "donor", "acceptor"]


def load(manifest: list[dict], jdir: Path, files: list[str]) -> pl.DataFrame:
    """Sum junction reads over the given BAM files (canonical-strand junctions only)."""
    frames = []
    for m in manifest:
        if m["file"] in files:
            frames.append(pl.read_parquet(jdir / f"{m['experiment']}_{m['file']}.parquet"))
    df = pl.concat(frames).filter(pl.col("strand") != "?")
    return df.group_by(J).agg(pl.col("reads").sum())


def site_usage(kd: pl.DataFrame, ctrl: pl.DataFrame) -> pl.DataFrame:
    """Per-site usage in knockdown and control over the union of observed junctions."""
    u = (kd.rename({"reads": "n_kd"}).join(ctrl.rename({"reads": "n_ctrl"}), on=J, how="full",
                                            coalesce=True)
         .with_columns(pl.col("n_kd").fill_null(0), pl.col("n_ctrl").fill_null(0)))
    out = []
    for site, partner in (("acceptor", "donor"), ("donor", "acceptor")):
        tot = u.group_by(["chrom", "strand", partner]).agg(
            pl.col("n_kd").sum().alias("tot_kd"), pl.col("n_ctrl").sum().alias("tot_ctrl"))
        s = (u.join(tot, on=["chrom", "strand", partner])
             .group_by(["chrom", "strand", site])
             .agg(pl.col("n_kd").sum().alias("num_kd"), pl.col("tot_kd").sum().alias("den_kd"),
                  pl.col("n_ctrl").sum().alias("num_ctrl"),
                  pl.col("tot_ctrl").sum().alias("den_ctrl"))
             .rename({site: "position"}).with_columns(pl.lit(site).alias("splice_type")))
        out.append(s)
    s = pl.concat(out)
    return s.with_columns(
        (pl.col("num_kd") / pl.col("den_kd")).alias("usage_kd"),
        (pl.col("num_ctrl") / pl.col("den_ctrl")).alias("usage_ctrl"),
    ).with_columns((pl.col("usage_kd") - pl.col("usage_ctrl")).alias("delta"))


def induced_sites(u: pl.DataFrame, min_den: int) -> pl.DataFrame:
    """Sites unused in control (usage <= 0.01) and used after knockdown (>= 0.10, >= 5 reads)."""
    return u.filter((pl.col("den_ctrl") >= min_den) & (pl.col("den_kd") >= min_den)
                    & (pl.col("usage_ctrl") <= 0.01) & (pl.col("usage_kd") >= 0.10)
                    & (pl.col("num_kd") >= 5))


def noise_floor(manifest: list[dict], jdir: Path, min_den: int) -> dict[str, list[int]]:
    """Induced-site counts between two replicates of the same control experiment, both ways."""
    out = {}
    for exp in sorted({m["experiment"] for m in manifest if m["role"] == "control"}):
        reps = sorted(m["file"] for m in manifest if m["experiment"] == exp)
        if len(reps) < 2:
            continue
        a, b = load(manifest, jdir, [reps[0]]), load(manifest, jdir, [reps[1]])
        out[exp] = [induced_sites(site_usage(a, b), min_den).height,
                    induced_sites(site_usage(b, a), min_den).height]
    return out


def event_dpsi(kd: pl.DataFrame, ctrl: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Long-form ΔPSI of A3SS/A5SS events from junction reads alone."""
    def psi(j: pl.DataFrame, tag: str) -> pl.DataFrame:
        e = events
        a3 = e.filter(pl.col("event_type") == "A3SS")
        a5 = e.filter(pl.col("event_type") == "A5SS")
        def look(df, d, a, name):
            return df.join(j.rename({"donor": d, "acceptor": a, "reads": name}),
                           on=["chrom", "strand", d, a], how="left")
        a3 = look(look(a3.with_columns(pl.col("partner_site").alias("_d")), "_d", "long_site", "nl"),
                  "_d", "short_site", "ns")
        a5 = look(look(a5.with_columns(pl.col("partner_site").alias("_a")), "long_site", "_a", "nl"),
                  "short_site", "_a", "ns")
        both = pl.concat([a3.drop("_d"), a5.drop("_a")], how="diagonal_relaxed")
        return both.with_columns(pl.col("nl").fill_null(0), pl.col("ns").fill_null(0)).with_columns(
            (pl.col("nl") + pl.col("ns")).alias(f"reads_{tag}"),
            (pl.col("nl") / (pl.col("nl") + pl.col("ns"))).alias(f"psi_{tag}")).drop("nl", "ns")
    k = psi(kd, "kd").select("event_key", "dpsi", "fdr", "reads_kd", "psi_kd")
    c = psi(ctrl, "ctrl").select("event_key", "reads_ctrl", "psi_ctrl")
    return k.join(c, on="event_key").with_columns(
        (pl.col("psi_kd") - pl.col("psi_ctrl")).alias("dpsi_junction"))


def annotated_sites() -> pl.DataFrame:
    frames = []
    for name, path in ANNOT.items():
        if path.exists():
            frames.append(pl.read_csv(path, separator="\t", columns=["chrom", "position", "strand",
                                                                      "splice_type"],
                                      schema_overrides={"chrom": pl.String})
                          .with_columns(pl.col("chrom").str.replace(r"^chr", ""),
                                        pl.lit(name).alias("source")))
    return pl.concat(frames).unique()


_CTX: dict = {}


def _init() -> None:
    """Load the inputs every target needs once per worker process."""
    _CTX["events"] = pl.read_parquet(EVENTS, columns=["event_key", "rbp", "event_type", "chrom",
                                                      "strand", "long_site", "short_site",
                                                      "partner_site", "dpsi", "fdr"])
    _CTX["annot"] = annotated_sites()
    _CTX["genes"] = pl.read_csv(GENES, separator="\t", columns=["gene_name", "chrom", "start", "end"]
                                ).with_columns(pl.col("chrom").str.replace(r"^chr", ""))


def check_target(target: str, manifest: list[dict], jdir: Path, min_den: int,
                 null: dict[str, list[int]]) -> tuple[str, dict, str]:
    """Checks 2-5 for one knockdown; returns (target, report entry, printable summary)."""
    kd_rows = [m for m in manifest if m["role"] == "kd" and m["target"] == target]
    kd_files = [m["file"] for m in kd_rows]
    ctrl_exps = {e for m in kd_rows for e in m.get("control_experiments", [])}
    ctrl_files = [m["file"] for m in manifest if m["role"] == "control"
                  and m["experiment"] in ctrl_exps]
    kd, ctrl = load(manifest, jdir, kd_files), load(manifest, jdir, ctrl_files)
    res: dict = {"kd_files": kd_files, "control_files": ctrl_files}

    # 2-3. agreement with rMATS on this regulator's events
    ev = event_dpsi(kd, ctrl, _CTX["events"].filter(pl.col("rbp") == target))
    cov = ev.filter((pl.col("reads_kd") >= min_den) & (pl.col("reads_ctrl") >= min_den))
    sig = cov.filter((pl.col("fdr") < 0.05) & (pl.col("dpsi").abs() >= 0.1))
    if cov.height > 2:
        res["rmats_events_covered"] = cov.height
        res["rmats_pearson"] = round(float(np.corrcoef(cov["dpsi"], cov["dpsi_junction"])[0, 1]), 3)
    if sig.height:
        res["rmats_sign_agreement_sig"] = round(float(
            (np.sign(sig["dpsi"]) == np.sign(sig["dpsi_junction"])).mean()), 3)
        res["rmats_sig_events"] = sig.height

    # replicate agreement: each knockdown replicate vs pooled control
    if len(kd_files) >= 2:
        r1, r2 = (site_usage(load(manifest, jdir, [f]), ctrl) for f in kd_files[:2])
        rr = r1.join(r2, on=["chrom", "strand", "position", "splice_type"], suffix="_r2").filter(
            (pl.col("den_kd") >= min_den) & (pl.col("den_kd_r2") >= min_den)
            & (pl.col("den_ctrl") >= min_den) & (pl.col("den_ctrl_r2") >= min_den)
            & ((pl.col("delta").abs() > 0.05) | (pl.col("delta_r2").abs() > 0.05)))
        if rr.height > 2:
            res["replicate_pearson_changed_sites"] = round(float(
                np.corrcoef(rr["delta"], rr["delta_r2"])[0, 1]), 3)
            res["replicate_sites"] = rr.height

    # 4. induced sites
    u = site_usage(kd, ctrl)
    induced = induced_sites(u, min_den).join(_CTX["annot"].group_by("chrom", "position", "strand", "splice_type")
                           .agg(pl.col("source").sort().str.join("+").alias("annotated_in")),
                           on=["chrom", "position", "strand", "splice_type"], how="left"
                           ).with_columns(pl.col("annotated_in").fill_null("none"))
    res["sites_measured"] = u.filter((pl.col("den_ctrl") >= min_den)
                                     & (pl.col("den_kd") >= min_den)).height
    res["induced_sites"] = induced.height
    res["induced_by_annotation"] = dict(induced.group_by("annotated_in").len().iter_rows())
    top = (induced.sort("delta", descending=True).head(8)
           .join_where(_CTX["genes"], pl.col("chrom") == pl.col("chrom_right"),
                       pl.col("position") >= pl.col("start"), pl.col("position") <= pl.col("end"))
           .select("gene_name", "chrom", "position", "strand", "splice_type", "usage_ctrl",
                   "usage_kd", "num_kd", "annotated_in").unique(["chrom", "position", "splice_type"]))
    res["top_induced"] = top.to_dicts()

    # 5. excess over the noise floor, at one replicate's depth on both sides
    ctrl1 = load(manifest, jdir, sorted(ctrl_files)[:1])
    single = [induced_sites(site_usage(load(manifest, jdir, [f]), ctrl1),
                            min_den).height for f in kd_files]
    floor = [n for e in sorted(ctrl_exps) for n in null.get(e, [])]
    res["induced_single_rep"] = single
    res["noise_floor_same_controls"] = floor
    if floor and np.mean(floor) > 0:
        res["excess_over_noise"] = round(float(np.mean(single) / np.mean(floor)), 2)

    text = (f"\n{target}: rMATS agreement r={res.get('rmats_pearson')} on {res.get('rmats_events_covered')} "
          f"covered events, sign agreement {res.get('rmats_sign_agreement_sig')} on "
          f"{res.get('rmats_sig_events')} significant | replicate r={res.get('replicate_pearson_changed_sites')} "
          f"| induced sites {res['induced_sites']} of {res['sites_measured']:,} measured "
          f"{res['induced_by_annotation']} | excess over noise {res.get('excess_over_noise')} "
          f"({single} vs {floor})")
    with pl.Config(tbl_rows=8, float_precision=3, tbl_width_chars=160):
        text += "\n" + str(top)
    return target, res, text



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--junction-dir", type=Path, default=JUNC)
    ap.add_argument("--min-den", type=int, default=20, help="min partner reads for a usage")
    ap.add_argument("--targets", nargs="+", help="knockdowns to check (default: all counted)")
    ap.add_argument("--workers", type=int, default=1, help="knockdowns checked in parallel")
    ap.add_argument("--report", type=Path, help="output JSON (default: <junction-dir>/pilot_report.json)")
    args = ap.parse_args()

    manifest = [m for m in json.loads((args.junction_dir / "manifest.json").read_text())
                if m.get("status") in ("ok", "exists")]
    report: dict = {"quality": [], "targets": {}}
    null = noise_floor(manifest, args.junction_dir, args.min_den)
    report["noise_floor"] = null

    print("1. Quality per BAM")
    for m in sorted(manifest, key=lambda m: (m["role"], m["target"], m["replicate"])):
        q = {k: m.get(k) for k in ("role", "target", "file", "replicate", "reads", "spliced_reads",
                                   "junctions", "junction_reads", "canonical_read_frac")}
        report["quality"].append(q)
        print(f"   {q['role']:7s} {q['target']:8s} rep {q['replicate']}: "
              f"{(q['reads'] or 0) / 1e6:.0f}M reads, {(q['spliced_reads'] or 0) / 1e6:.1f}M spliced, "
              f"{q['junctions']:,} junctions, canonical {q['canonical_read_frac']}")

    print(f"\n5. Noise floor (control rep vs control rep, both ways): {null}")
    targets = sorted({m["target"] for m in manifest if m["role"] == "kd"})
    selected = [t for t in targets if not args.targets or t in args.targets]
    check = partial(check_target, manifest=manifest, jdir=args.junction_dir, min_den=args.min_den,
                    null=null)
    if args.workers <= 1:
        _init()
        results = [check(t) for t in selected]
    else:
        os.environ.setdefault("POLARS_MAX_THREADS", "2")  # many processes, few threads each
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init) as ex:
            results = list(ex.map(check, selected))
    for target, res, text in results:
        report["targets"][target] = res
        print(text)

    out = args.report or args.junction_dir / "pilot_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

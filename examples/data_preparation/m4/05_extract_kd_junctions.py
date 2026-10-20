#!/usr/bin/env python
"""Count splice-junction reads in ENCODE knockdown and control RNA-seq (position-level M4).

The event-level M4 corpus comes from rMATS tables, which only describe pre-defined A3SS/A5SS
events. A position-level M4 needs, for every splice site, how many reads use it in knockdown and
in control, including sites that are unused normally and switch on after a knockdown. Those counts
come from the read alignments themselves. This script, for each requested knockdown:

1. asks the ENCODE portal for the experiment and its linked control experiment;
2. picks one GRCh38 genome-alignment BAM per biological replicate (the most recent);
3. downloads it (md5-checked), counts reads that cross each junction, and deletes the BAM.

A read counts toward a junction (an ``N`` in its CIGAR) when it is a primary, uniquely mapped
(MAPQ 255, STAR's unique code) alignment with at least ``--min-anchor`` aligned bases on both
sides. Mates are counted separately. Junction strand comes from the intron motif in the reference
(GT-AG or GC-AG on the plus strand, their reverse complements on the minus strand), as STAR does;
other motifs are kept with strand ``?``.

Coordinates follow the rest of the M4 corpus: 1-based, and each junction also carries its
``donor`` and ``acceptor`` as the exonic base at each splice boundary (on the plus strand
``donor = intron_start - 1`` and ``acceptor = intron_end + 1``; mirrored on the minus strand).

Output (``--out-dir``): one parquet per BAM (``<experiment>_<file>.parquet``: chrom, intron_start,
intron_end, strand, motif, donor, acceptor, reads) and ``manifest.json`` (experiment, target,
role, replicate, file, md5, read and junction counts).

Run on a pod (BAMs are ~4 GB each):
    python examples/data_preparation/m4/05_extract_kd_junctions.py \\
        --targets U2AF1 U2AF2 QKI PTBP1 TARDBP SF3B1 --cell-line HepG2 \\
        --tmp-dir /runpod-volume/tmp/bams --workers 6

All HepG2 knockdowns (232 proteins + 26 shared controls, ~516 BAMs; reruns skip counted files):
    python examples/data_preparation/m4/05_extract_kd_junctions.py --targets all \\
        --cell-line HepG2 --tmp-dir /runpod-volume/tmp/bams --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
OUT = REPO / "data/mane/GRCh38/m4_junctions"
PORTAL = "https://www.encodeproject.org"

# CIGAR operation codes (pysam): M, I, D, N, S, H, P, =, X
_M, _I, _D, _N, _S, _H, _P, _EQ, _X = range(9)
_REF_ALIGNED = (_M, _EQ, _X)  # consume reference and query
PLUS_MOTIFS = {("GT", "AG"): "GT-AG", ("GC", "AG"): "GC-AG", ("AT", "AC"): "AT-AC"}
MINUS_MOTIFS = {("CT", "AC"): "GT-AG", ("CT", "GC"): "GC-AG", ("GT", "AT"): "AT-AC"}


# ── ENCODE portal ────────────────────────────────────────────────────────


def _get(path: str) -> dict:
    import requests

    r = requests.get(PORTAL + path, headers={"Accept": "application/json"}, timeout=120)
    r.raise_for_status()
    return r.json()


def _acc(x) -> str:
    if isinstance(x, dict):
        return x.get("accession") or x["@id"].rstrip("/").split("/")[-1]
    return x.rstrip("/").split("/")[-1]


def genome_bams(experiment: str) -> list[dict]:
    """One released GRCh38 genome-alignment BAM per biological replicate (most recent)."""
    exp = _get(f"/experiments/{experiment}/?frame=embedded&format=json")
    files = [f for f in exp.get("files", []) if isinstance(f, dict)
             and f.get("output_type") == "alignments" and f.get("file_format") == "bam"
             and f.get("assembly") == "GRCh38" and f.get("status") == "released"]
    best: dict[tuple, dict] = {}
    for f in files:
        rep = tuple(f.get("biological_replicates") or [0])
        if rep not in best or f.get("date_created", "") > best[rep].get("date_created", ""):
            best[rep] = f
    return [{"experiment": experiment, "file": f["accession"], "href": f["href"],
             "md5": f.get("md5sum"), "size": f.get("file_size"),
             "replicate": ",".join(map(str, rep))} for rep, f in sorted(best.items())]


def all_targets(cell_line: str) -> list[str]:
    """Every protein with a released shRNA RNA-seq knockdown in the cell line."""
    q = (f"/search/?type=Experiment&assay_title=shRNA+RNA-seq&biosample_ontology.term_name="
         f"{cell_line}&status=released&frame=object&format=json&limit=all&field=target")
    hits = _get(q).get("@graph", [])
    return sorted({_acc(e["target"]).rsplit("-", 1)[0] for e in hits if e.get("target")})


def plan_downloads(targets: list[str], cell_line: str) -> list[dict]:
    """Knockdown and control BAMs for each target (controls deduplicated across targets)."""
    jobs, seen = [], set()
    for t in targets:
        q = (f"/search/?type=Experiment&assay_title=shRNA+RNA-seq&target.label={t}"
             f"&biosample_ontology.term_name={cell_line}&status=released&frame=object&format=json")
        hits = _get(q).get("@graph", [])
        if not hits:
            print(f"  {t}: no released {cell_line} shRNA RNA-seq experiment; skipped")
            continue
        exp = hits[0]
        ctrls = [_acc(c) for c in exp.get("possible_controls", [])]
        for b in genome_bams(exp["accession"]):
            jobs.append({**b, "target": t, "role": "kd", "control_experiments": ctrls})
        for c in ctrls:  # a control shared by several knockdowns is downloaded once
            for b in genome_bams(c):
                if b["file"] in seen:
                    continue
                seen.add(b["file"])
                jobs.append({**b, "target": "control", "role": "control",
                             "control_experiments": []})
        print(f"  {t}: knockdown {exp['accession']}, control {', '.join(ctrls)}")
    return jobs


# ── Junction counting ────────────────────────────────────────────────────


def count_junctions(bam_path: str, min_anchor: int, mapq: int) -> tuple[Counter, dict]:
    """Count (chrom, intron_start0, intron_end0) over reads passing the filters."""
    import pysam

    junctions: Counter = Counter()
    stats = Counter()
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            stats["reads"] += 1
            if (read.is_unmapped or read.is_secondary or read.is_supplementary
                    or read.is_qcfail or read.is_duplicate or read.mapping_quality < mapq):
                continue
            cig = read.cigartuples
            if not cig or not any(op == _N for op, _ in cig):
                continue
            stats["spliced_reads"] += 1
            pos = read.reference_start
            blocks: list[tuple[str, int, int]] = []  # ("A" aligned length | "N" intron start,len)
            for op, length in cig:
                if op in _REF_ALIGNED:
                    blocks.append(("A", length, pos))
                    pos += length
                elif op == _D:
                    pos += length
                elif op == _N:
                    blocks.append(("N", length, pos))
                    pos += length
            for i, (kind, length, start) in enumerate(blocks):
                if kind != "N":
                    continue
                left = blocks[i - 1][1] if i > 0 and blocks[i - 1][0] == "A" else 0
                right = blocks[i + 1][1] if i + 1 < len(blocks) and blocks[i + 1][0] == "A" else 0
                if left >= min_anchor and right >= min_anchor:
                    junctions[(read.reference_name, start, start + length)] += 1
    return junctions, dict(stats)


def to_table(junctions: Counter, fasta_path: Path) -> pl.DataFrame:
    """Junction counts with strand from the intron motif and donor/acceptor exonic bases."""
    import pyfaidx

    fa = pyfaidx.Fasta(str(fasta_path), sequence_always_upper=True, rebuild=False)
    rows = []
    for (chrom, s0, e0), n in junctions.items():
        c = chrom[3:] if chrom.startswith("chr") else chrom
        if c not in fa:
            continue
        seq = fa[c]
        first, last = str(seq[s0:s0 + 2]), str(seq[e0 - 2:e0])
        strand, motif = "?", f"{first}-{last}"
        if (first, last) in PLUS_MOTIFS:
            strand, motif = "+", PLUS_MOTIFS[(first, last)]
        elif (first, last) in MINUS_MOTIFS:
            strand, motif = "-", MINUS_MOTIFS[(first, last)]
        istart, iend = s0 + 1, e0  # 1-based, inclusive intron
        donor, acceptor = (istart - 1, iend + 1) if strand != "-" else (iend + 1, istart - 1)
        rows.append((c, istart, iend, strand, motif, donor, acceptor, n))
    return pl.DataFrame(rows, schema={"chrom": pl.String, "intron_start": pl.Int64,
                                      "intron_end": pl.Int64, "strand": pl.String,
                                      "motif": pl.String, "donor": pl.Int64,
                                      "acceptor": pl.Int64, "reads": pl.Int64}, orient="row")


def process(job: dict, tmp_dir: str, out_dir: str, min_anchor: int, mapq: int) -> dict:
    """Download one BAM, check md5, count junctions, write the table, delete the BAM."""
    import requests

    out_path = Path(out_dir) / f"{job['experiment']}_{job['file']}.parquet"
    if out_path.exists():
        return {**job, "status": "exists"}
    bam = Path(tmp_dir) / f"{job['file']}.bam"
    t0, md5 = time.time(), hashlib.md5()
    try:
        with requests.get(PORTAL + job["href"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(bam, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8 << 20):
                    fh.write(chunk)
                    md5.update(chunk)
    except Exception:
        bam.unlink(missing_ok=True)  # never leave a partial BAM on the volume
        raise
    if job.get("md5") and md5.hexdigest() != job["md5"]:
        bam.unlink(missing_ok=True)
        raise RuntimeError(f"md5 mismatch for {job['file']}")
    t_dl = time.time() - t0
    junctions, stats = count_junctions(str(bam), min_anchor, mapq)
    bam.unlink(missing_ok=True)
    table = to_table(junctions, FASTA)
    tmp = out_path.with_suffix(".tmp")
    table.write_parquet(tmp)
    tmp.replace(out_path)
    canon = table.filter(pl.col("strand") != "?")["reads"].sum() / max(1, table["reads"].sum())
    return {**job, "status": "ok", **stats, "junctions": table.height,
            "junction_reads": int(table["reads"].sum()), "canonical_read_frac": round(canon, 4),
            "download_s": round(t_dl), "total_s": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", required=True,
                    help="knocked-down protein names, or 'all' for every released knockdown")
    ap.add_argument("--cell-line", default="HepG2")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--tmp-dir", type=Path, required=True, help="scratch space for BAMs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-anchor", type=int, default=8)
    ap.add_argument("--mapq", type=int, default=255, help="255 = uniquely mapped (STAR)")
    ap.add_argument("--plan-only", action="store_true", help="list the BAMs and stop")
    args = ap.parse_args()

    if args.targets == ["all"]:
        args.targets = all_targets(args.cell_line)
    print(f"Planning {args.cell_line} downloads for {len(args.targets)} targets:")
    jobs = plan_downloads(args.targets, args.cell_line)
    gb = sum(j["size"] or 0 for j in jobs) / 1e9
    print(f"{len(jobs)} BAMs, {gb:.0f} GB total")
    if args.plan_only:
        for j in jobs:
            print(f"  {j['role']:7s} {j['target']:8s} {j['experiment']} rep {j['replicate']} "
                  f"{j['file']} {(j['size'] or 0) / 1e9:.1f} GB")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, j, str(args.tmp_dir), str(args.out_dir), args.min_anchor,
                          args.mapq): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 — record and continue with other files
                res = {**j, "status": f"failed: {exc}"}
            if res["status"] == "exists" and any(m.get("file") == res["file"]
                                                 and m.get("status") == "ok" for m in manifest):
                continue  # counted by an earlier run; keep its stats
            manifest = [m for m in manifest if m.get("file") != res["file"]] + [res]
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"  [{i}/{len(jobs)}] {res['role']} {res['target']} {res['file']}: "
                  f"{res['status']} {res.get('junctions', '')} junctions, "
                  f"canonical {res.get('canonical_read_frac', '')} ({res.get('total_s', '')}s)",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

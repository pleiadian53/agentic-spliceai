#!/usr/bin/env python
"""Phase E: warm the Bio Lab UI caches for a live meta-overlay demo.

Hits a RUNNING Bio Lab server's genome endpoint for each demo gene — base-only
and with each meta overlay — so the server's in-memory model + prediction caches
are hot and the first click in the live demo is instant. Doubles as an
end-to-end smoke test of the meta overlay (Phases B–D).

The report groups counts by **truth set**, because that is the thing most easily
misread here. Ground truth is resolved from the *meta* model's training
annotation, so one base prediction is scored twice: an M1-S block scores it
against MANE, an M2-S block against Ensembl. Base reading ``44/44`` in one block
and ``44/70`` in the next is the same prediction against two yardsticks, not two
predictions. A trailing legend explains the columns using a real gene from the
run.

Prerequisites
-------------
1. Phase-A feature cache exists for these genes (instant if already warmed):
     python examples/UI_integration/02_build_showcase_feature_cache.py
2. The server is running:
     ~/miniforge3/envs/agentic-spliceai/bin/python -m server.bio.app   # port 8005

Usage
-----
    PY=~/miniforge3/envs/agentic-spliceai/bin/python
    $PY examples/UI_integration/07_warm_ui_cache.py
    $PY examples/UI_integration/07_warm_ui_cache.py --genes UNC13A STMN2
    $PY examples/UI_integration/07_warm_ui_cache.py --no-legend
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# BRCA1 + TP53 + ALS panel — the showcase set built by
# 02_build_showcase_feature_cache.py, which must be kept in sync with this list.
# The meta calls below read the .npz that script writes, so a gene here without
# one gets a 404 rather than a warm cache.
#
# TP53 earns its place as the cleanest alternative-site contrast in the set: 20
# MANE sites against 35 in Ensembl, so the delta is large enough to see and the
# canonical calls are unambiguous.
DEFAULT_GENES = ["BRCA1", "TP53", "STMN2", "UNC13A", "SOD1", "TARDBP", "FUS", "C9orf72"]
# Canonical <variant>.<arch>.<corpus> keys. The retired spellings
# (m1s_v4_cleanannot, ...) still resolve server-side via META_MODEL_ALIASES, so
# passing either to --meta-models works.
DEFAULT_META = ["m1s.concat_fusion.cleanannot", "m2s.concat_fusion.cleanannot"]

#: Threshold used for the legend's "same prediction, sane threshold" comparison.
#: M2-S is F1-optimal near 0.99 on the held-out evaluation; the warming threshold
#: (0.5) is nobody's operating point, which is what makes its FP column alarming.
OPERATING_POINT = 0.99

#: Track key holding the alternative-site delta (Ensembl minus MANE).
ALT_TRACK = "ensembl_minus_mane"

# A THIRD category, and the one the annotation-based columns structurally cannot
# see. Alternative sites are normal biology that MANE simply omits, so a bigger
# annotation (Ensembl) finds them. Disease-induced cryptic sites are absent from
# MANE, Ensembl AND GENCODE — verified for all three ALS sites — because they are
# repressed in healthy tissue and *should not* be annotated as normal splice
# sites. Their absence is correct biology, not an annotation gap.
#
# The consequence for this report: a correct cryptic detection is a false
# positive against every truth set, by construction. On UNC13A the base model
# scores the ALS cryptic donor 0.000 and M2-S scores it 0.508, and that recovery
# lands inside the "146 FP" column. So the cryptic block below is reported as
# anchored recall against curated coordinates, never as TP/FP/FN.
#
# This is also why M3 (novel-site ranker) is scored by per-gene precision@k on
# independent truth, and why M4 (perturbation) is a separate variant at all.
try:
    from als_cryptic_sites import EVENTS as _CRYPTIC_EVENTS
except ImportError:  # registry moved or script imported from elsewhere
    _CRYPTIC_EVENTS = []
CRYPTIC_BY_GENE: dict[str, list[tuple[int, str]]] = {}
for _ev in _CRYPTIC_EVENTS:
    CRYPTIC_BY_GENE.setdefault(_ev.gene, []).extend((s.pos, s.kind) for s in _ev.sites)


def _get(url: str, timeout: float = 900.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _sites(markers, kind: str) -> set:
    """(position, splice_type) pairs of one prediction class from a marker list."""
    return {(m["position"], m["site_type"]) for m in markers if m["pred_type"] == kind}


def _alt_in_window(tracks: dict | None, start: int, end: int) -> set:
    """Alternative sites (Ensembl minus MANE) inside the scored gene window.

    Clipped to the window because prediction runs over the gene's span in the
    *model's* annotation, and Ensembl genes are frequently longer. Counting
    sites the model was never shown would understate every model.
    """
    if not tracks:
        return set()
    track = next((t for t in tracks.get("tracks", []) if t.get("key") == ALT_TRACK), None)
    if not track or not track.get("available"):
        return set()
    out = set()
    for kind in ("donor", "acceptor"):
        out |= {(p, kind) for p in track.get(kind, []) if start <= p <= end}
    return out


#: Peak-preserving downsample floor in bio_service._build_overlay_response. A
#: position absent from a response's arrays was below this in EVERY track of
#: that response, so "missing" carries information and is not a measurement gap.
PLOT_FLOOR = 0.05


def _cryptic_scores(responses: list, sites: list, meta_key: str | None) -> list:
    """Model score at each curated cryptic position, searching several responses.

    ``meta_key`` selects a meta model's arrays; ``None`` reads the base arrays,
    which are identical across responses, so any response carrying the position
    answers for it. Responses are downsampled independently, and a low-scoring
    position can be dropped from one while surviving in another, so looking in
    only one would report a real score as missing. Returns ``None`` only when no
    response has the position, which means every track there was below
    :data:`PLOT_FLOOR`.
    """
    out = []
    for pos, kind in sites:
        dk = "meta_donor_prob" if meta_key else "donor_prob"
        ak = "meta_acceptor_prob" if meta_key else "acceptor_prob"
        val = None
        for d in responses:
            if meta_key and d.get("meta_model") != meta_key:
                continue
            if "_idx" not in d:
                d["_idx"] = {p: i for i, p in enumerate(d["positions"])}
            i = d["_idx"].get(pos)
            if i is not None:
                val = d[ak if kind == "acceptor" else dk][i]
                break
        out.append(val)
    return out


def _fmt_cryptic(label: str, sites: list, scores: list, threshold: float) -> str:
    hit = sum(1 for s in scores if s is not None and s > threshold)
    detail = "  ".join(
        f"{kind[:3]}@{pos:,} {f'<{PLOT_FLOOR}' if s is None else f'{s:.3f}'}"
        for (pos, kind), s in zip(sites, scores, strict=True)
    )
    return f"    {label:<30s} {hit}/{len(sites)}   {detail}"


def _fmt_row(label: str, tp: int, n: int, fp: int, fn: int, extra: str = "",
             secs: float | None = None) -> str:
    row = f"    {label:<30s} TP {tp:>3}/{n:<3}  FP {fp:>4}  FN {fn:>3}{extra}"
    if secs is not None:
        row = f"{row}   ({secs:4.1f}s)"
    return row.rstrip()


def _legend(example: dict | None, threshold: float) -> str:
    """Column guide, anchored to a real gene from this run rather than prose.

    Uses measured numbers so it cannot drift from the code, and so a surprising
    line in the report above can be checked against a worked case.
    """
    lines = [
        "",
        "─" * 78,
        "How to read this",
        "",
        "  vs <truth>   The annotation the counts are scored against, resolved from the",
        "               META model's training annotation. The SAME base prediction is",
        "               re-scored per block, so never compare numbers across blocks.",
        "  TP x/N       N = every truth site inside the scored gene window (TP + FN).",
        "  alt x/M      M = Ensembl \\ MANE, the alternative sites. Finding these is",
        "               exactly what M2-S exists to do and what the base model cannot.",
        "  cryptic      Disease-induced sites, curated from the literature. Absent from",
        "               MANE, Ensembl AND GENCODE, because they are repressed in healthy",
        "               tissue and are not normal splice sites. Reported as anchored",
        "               recall + raw scores: a correct detection is an FP against every",
        "               truth set, so TP/FP/FN cannot express it. This is the gap M3",
        "               and M4 exist to fill.",
        f"  FP           Warming runs at threshold {threshold}, which is nobody's operating point.",
        f"               M2-S is F1-optimal near {OPERATING_POINT}. Read FP here as a smoke-test",
        "               signal, not a result.",
    ]
    if example:
        n, n_alt = example["n"], example["n_alt"]
        lines += [
            "",
            f"Worked example — {example['gene']}, vs ensembl:",
            f"  {n} truth sites in the window = {n - n_alt} canonical + {n_alt} alternative.",
            f"  Base finds the canonical ones but {example['base_alt']}/{n_alt} alternative,",
            f"  so its {example['base_fn']} misses ARE the alternative sites.",
            f"  {example['meta']} recovers {example['meta_alt']}/{n_alt} of them:",
            f"  misses {example['base_fn']} → {example['meta_fn']}.",
        ]
        if example.get("op") is not None:
            op = example["op"]
            lines += [
                f"  Its {example['meta_fp']} false positives are a threshold artifact. The same",
                f"  prediction at {OPERATING_POINT}: FP {op['meta_fp']}, TP {op['meta_tp']}/{n}.",
            ]
    lines.append("─" * 78)
    return "\n".join(lines)


def _names(values: list[str]) -> list[str]:
    """Flatten a repeated argument that may also be comma-separated.

    ``--genes TP53 SOD1`` and ``--genes TP53,SOD1`` both work. Without this the
    comma form is accepted silently as a single name and fails much later as an
    unhelpful 404 on a gene called "TP53,SOD1".
    """
    out: list[str] = []
    for v in values:
        out.extend(part.strip() for part in v.split(",") if part.strip())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Warm + smoke-test the Bio Lab meta overlay")
    p.add_argument("--server", default="http://localhost:8005")
    p.add_argument("--genes", nargs="+", default=DEFAULT_GENES,
                   help="Space- or comma-separated gene symbols")
    p.add_argument("--base-model", default="openspliceai")
    p.add_argument("--meta-models", nargs="+", default=DEFAULT_META,
                   help="Space- or comma-separated meta model keys")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-legend", action="store_true",
                   help="Skip the trailing column guide")
    args = p.parse_args()
    base = args.server.rstrip("/")
    args.genes = _names(args.genes)
    args.meta_models = _names(args.meta_models)

    print(f"Warming {base} for {len(args.genes)} genes × ({args.base_model} + "
          f"{', '.join(args.meta_models)}) @ threshold {args.threshold}\n")
    ok = fail = 0
    base_failures: list[str] = []
    meta_failures: list[str] = []
    example: dict | None = None

    for gene in args.genes:
        g = urllib.parse.quote(gene)
        print(gene)

        # Base-only first: it establishes the window the tracks get clipped to,
        # and its truth set is the default (the base model's own annotation).
        try:
            t = time.time()
            d0 = _get(f"{base}/api/genome/{g}/predict"
                      f"?model={args.base_model}&threshold={args.threshold}")
            ok += 1
            elapsed = {(d0.get("truth"), "base"): time.time() - t}
        except Exception as e:  # noqa: BLE001
            print(f"    base FAILED: {e}\n")
            fail += 1
            base_failures.append(gene)
            continue

        start, end = d0["gene_start"], d0["gene_end"]
        try:
            tracks = _get(f"{base}/api/genome/{g}/annotation-tracks"
                          f"?chrom={urllib.parse.quote(str(d0['chrom']))}", timeout=120.0)
        except Exception:  # noqa: BLE001
            tracks = None  # additive context only; never fail the warm-up over it
        alt = _alt_in_window(tracks, start, end)

        # Collect every (truth set -> rows) before printing, so one base row can
        # head each block instead of repeating per meta model.
        blocks: dict[str, dict] = {}
        blocks[d0.get("truth", "?")] = {"base": d0, "metas": []}

        for mm in args.meta_models:
            try:
                t = time.time()
                d = _get(f"{base}/api/genome/{g}/predict?model={args.base_model}"
                         f"&meta={urllib.parse.quote(mm)}&threshold={args.threshold}")
                ok += 1
                elapsed[(d.get("truth"), mm)] = time.time() - t
                blocks.setdefault(d.get("truth", "?"), {"base": d, "metas": []})
                blocks[d.get("truth", "?")]["metas"].append((mm, d))
            except Exception as e:  # noqa: BLE001
                print(f"    ↳ {mm} FAILED: {e}")
                fail += 1
                meta_failures.append(gene)

        for truth, blk in blocks.items():
            b = blk["base"]
            n = b["n_tp"] + b["n_fn"]
            n_alt = len(alt)
            # The delta is a subset of Ensembl/GENCODE, never of MANE, so the
            # composition line only means something on those truth sets.
            has_alt = n_alt > 0 and truth != "mane"
            comp = (f" = {n - n_alt} canonical + {n_alt} alternative" if has_alt else "")
            print(f"  vs {truth} · {n} sites{comp}")

            b_tp = _sites(b["markers"], "TP")
            print(_fmt_row("base", b["n_tp"], n, b["n_fp"], b["n_fn"],
                           f"  alt {len(b_tp & alt):>3}/{n_alt}" if has_alt else "",
                           elapsed.get((truth, "base"))))

            for mm, d in blk["metas"]:
                m_tp = _sites(d["meta_markers"], "TP")
                extra = f"  alt {len(m_tp & alt):>3}/{n_alt}" if has_alt else ""
                if not has_alt and alt:
                    # On a MANE block the alternative sites are not truth, so a
                    # call on one lands in FP. Saying how many explains the
                    # count instead of leaving it looking like noise.
                    hit = len(_sites(d["meta_markers"], "FP") & alt)
                    if hit:
                        extra = f"  ({hit}/{d['meta_n_fp']} FP are alt sites)"
                print(_fmt_row(mm, d["meta_n_tp"], n, d["meta_n_fp"], d["meta_n_fn"], extra,
                               elapsed.get((truth, mm))))

                if example is None and has_alt:
                    example = {
                        "gene": gene, "meta": mm, "n": n, "n_alt": n_alt,
                        "base_alt": len(b_tp & alt), "meta_alt": len(m_tp & alt),
                        "base_fn": b["n_fn"], "meta_fn": d["meta_n_fn"],
                        "meta_fp": d["meta_n_fp"], "op": None,
                    }

        # Curated cryptic sites, if this gene has any. Reported as anchored
        # recall with raw scores, because TP/FP/FN is undefined here: these
        # positions are in no annotation, so every detection above counted them
        # as a false positive.
        crypt = CRYPTIC_BY_GENE.get(gene)
        meta_rows = [r for blk in blocks.values() for r in blk["metas"]]
        if crypt and meta_rows:
            plural = "" if len(crypt) == 1 else "s"
            print(f"  cryptic · {len(crypt)} site{plural} (disease-induced; in NO annotation, "
                  f"so hits land in FP above)")
            responses = [d0] + [d for _, d in meta_rows]
            print(_fmt_cryptic("base", crypt,
                               _cryptic_scores(responses, crypt, None), args.threshold))
            for mm, _d in meta_rows:
                print(_fmt_cryptic(mm, crypt,
                                   _cryptic_scores(responses, crypt, mm), args.threshold))
        print()

    # One extra call so the legend's threshold claim is measured, not remembered.
    if example and not args.no_legend:
        try:
            d = _get(f"{base}/api/genome/{urllib.parse.quote(example['gene'])}/predict"
                     f"?model={args.base_model}&meta={urllib.parse.quote(example['meta'])}"
                     f"&threshold={args.threshold}&meta_threshold={OPERATING_POINT}")
            example["op"] = {"meta_fp": d["meta_n_fp"], "meta_tp": d["meta_n_tp"]}
        except Exception:  # noqa: BLE001
            pass

    print(f"Warmed {ok} responses, {fail} failed.")

    # Which hint depends on WHICH request failed. A base-path failure is the
    # server or the gene symbol; the feature cache is not involved, since the
    # base path never reads a multimodal channel. Pointing at
    # 02_build_showcase_feature_cache.py for a mistyped gene sends you to build
    # an .npz that would not have helped.
    if base_failures:
        print(f"\nBase prediction failed for: {', '.join(base_failures)}")
        print("  A base prediction needs no prebuilt data, so this is the server or the name.")
        print(f"  Check the server:  curl -s -o /dev/null -w '%{{http_code}}' {base}/")
        print("  Check the symbol:  the Gene Browser search box, or "
              f"{base}/api/genes?model={args.base_model}&search=<name>")
        print("  Note --genes takes several names: --genes TP53 SOD1  (or TP53,SOD1).")
    if meta_failures:
        print(f"\nMeta overlay failed for: {', '.join(sorted(set(meta_failures)))}")
        print("  The overlay is the only consumer of the on-disk feature cache. Build it with:")
        print(f"    python examples/UI_integration/02_build_showcase_feature_cache.py "
              f"--genes {' '.join(sorted(set(meta_failures)))}")
    if not args.no_legend:
        print(_legend(example, args.threshold))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

"""Local cache for the remote bigWig tracks the feature extractor streams.

Only the **conservation** modality (phyloP + phastCons) is fetched over the
network at feature-extraction time. Epigenetic marks resolve to local ENCODE
files and junction evidence to a local parquet, so this module deliberately
covers conservation alone rather than "all external tracks".

Why it exists: with no local copy, a DNS blip or an offline laptop makes
``pyBigWig`` return an error per region. That failure is now *loud* — the
extractor raises :class:`~.dense_feature_extractor.ChannelExtractionError` and
``build_gene_cache`` skips the gene rather than caching zeros — but a skipped
gene is still a gene you do not have. Caching the tracks locally removes the
network from the extraction path entirely, so the failure stops happening
rather than merely being reported.

(Before that fix the channel was silently zero-filled and written to disk; the
all-zero guard in ``sequence_level_dataset`` pools evidence across genes, so one
poisoned gene among healthy ones went undetected, including on resume.)

Sizes are large (~15.8 GB for GRCh38), so nothing here downloads without an
explicit decision by the caller.

Usage::

    from agentic_spliceai.splice_engine.features.track_cache import (
        missing_conservation_tracks, ensure_conservation_tracks,
    )

    ensure_conservation_tracks(build="GRCh38", download=True)
"""

from __future__ import annotations

import logging
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
_CHUNK = 8 * 1024 * 1024  # 8 MB


def _settings() -> dict:
    with open(_SETTINGS) as f:
        return yaml.safe_load(f) or {}


def default_cache_dir() -> Path:
    """``cache.bigwig_dir`` from settings.yaml, resolved against the project root.

    Unlike the extractor's private resolver this returns the path whether or not
    it exists — callers here need to be able to *create* it.
    """
    from agentic_spliceai.splice_engine.config.genomic_config import get_project_root

    rel = (_settings().get("cache") or {}).get("bigwig_dir", "data/cache/bigwig")
    return get_project_root() / rel


def conservation_tracks(build: str = "GRCh38") -> list[dict]:
    """The conservation bigWigs for *build*: ``{name, url, filename}``."""
    tracks = (_settings().get("external_tracks") or {}).get("conservation") or {}
    per_build = tracks.get(build) or {}
    out = []
    for name, spec in per_build.items():
        if isinstance(spec, dict) and spec.get("url") and spec.get("filename"):
            out.append({"name": name, "url": spec["url"], "filename": spec["filename"]})
    return out


def _remote_size(url: str, timeout: int = 30) -> int | None:
    """Content-Length for *url*, or None if the server will not say."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception as e:
        logger.debug("HEAD failed for %s: %s", url, e)
        return None


def missing_conservation_tracks(
    build: str = "GRCh38", cache_dir: Path | None = None
) -> list[dict]:
    """Conservation tracks absent from the cache, or present but truncated.

    A short file is reported as missing: a partially-downloaded bigWig is worse
    than none, because it opens successfully and returns wrong data for the
    regions past the truncation point.
    """
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    missing = []
    for t in conservation_tracks(build):
        local = cache_dir / t["filename"]
        expected = _remote_size(t["url"])
        if not local.exists():
            missing.append({**t, "reason": "absent", "bytes": expected})
        elif expected and local.stat().st_size != expected:
            missing.append({
                **t, "reason": f"incomplete ({local.stat().st_size:,} of {expected:,} bytes)",
                "bytes": expected,
            })
    return missing


def _download(url: str, dest: Path, expected: int | None) -> None:
    """Download *url* to *dest*, resuming a partial file when possible."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    if expected and have > expected:  # a stale .part from a different file
        tmp.unlink()
        have = 0

    req = urllib.request.Request(url)
    mode = "wb"
    if have:
        req.add_header("Range", f"bytes={have}-")
        mode = "ab"
        logger.info("Resuming %s at %.2f GB", dest.name, have / 1e9)

    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, mode) as f:
        # A server that ignores Range replies 200 and restarts the body.
        if have and r.status == 200:
            f.close()
            tmp.unlink()
            return _download(url, dest, expected)
        done = have
        last = -1
        while chunk := r.read(_CHUNK):
            f.write(chunk)
            done += len(chunk)
            if expected:
                pct = int(100 * done / expected)
                if pct >= last + 5:
                    logger.info("  %s %d%% (%.2f/%.2f GB)",
                                dest.name, pct, done / 1e9, expected / 1e9)
                    last = pct

    size = tmp.stat().st_size
    if expected and size != expected:
        raise OSError(
            f"{dest.name}: got {size:,} bytes, expected {expected:,}. "
            f"Partial file left at {tmp} — re-run to resume."
        )
    # Only becomes the real filename once complete, so an interrupted run can
    # never leave something the extractor would open and trust.
    tmp.replace(dest)
    logger.info("Downloaded %s (%.2f GB)", dest.name, size / 1e9)


def ensure_conservation_tracks(
    build: str = "GRCh38",
    cache_dir: Path | None = None,
    download: bool | str = "ask",
) -> Path | None:
    """Make sure the conservation bigWigs are cached locally.

    Parameters
    ----------
    download:
        ``True`` download without asking · ``False`` never download (the caller
        falls back to streaming) · ``"ask"`` prompt when attached to a terminal,
        otherwise print instructions and return ``None``.

    Returns
    -------
    The cache directory when every track is present, else ``None``.
    """
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    missing = missing_conservation_tracks(build, cache_dir)
    if not missing:
        return cache_dir

    total = sum(m["bytes"] or 0 for m in missing)
    lines = [f"    {m['filename']}  ({(m['bytes'] or 0) / 1e9:.2f} GB, {m['reason']})"
             for m in missing]
    summary = (
        f"  Conservation tracks not cached at {cache_dir}:\n"
        + "\n".join(lines)
        + f"\n  Total to download: {total / 1e9:.2f} GB"
    )

    if download is False:
        logger.warning(
            "%s\n  Continuing WITHOUT a local cache — conservation will stream from "
            "UCSC, and a network failure will zero-fill the channel.", summary
        )
        return None

    if download == "ask":
        if not sys.stdin.isatty():
            print(summary)
            print("  Re-run with --download-tracks to fetch them (non-interactive shell).")
            return None
        print(summary)
        if input("  Download now? [y/N] ").strip().lower() not in ("y", "yes"):
            print("  Skipped — conservation will stream from UCSC.")
            return None

    free = shutil.disk_usage(cache_dir.parent if cache_dir.exists()
                             else cache_dir.parent.parent).free
    if total and free < total * 1.1:
        raise OSError(
            f"Not enough free space: need ~{total / 1e9:.1f} GB, have {free / 1e9:.1f} GB."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for m in missing:
        logger.info("Fetching %s from %s", m["filename"], m["url"])
        _download(m["url"], cache_dir / m["filename"], m["bytes"])

    still = missing_conservation_tracks(build, cache_dir)
    if still:
        raise OSError(f"Still missing after download: {[m['filename'] for m in still]}")
    return cache_dir

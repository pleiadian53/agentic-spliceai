# Getting Started

## Launch

```bash
conda run -n agentic-spliceai python -m server.bio.app
# → http://localhost:8005/
```

The server reloads on changes under `server/bio/`. Starting a second one fails with
`[Errno 48] Address already in use`, so the entry point takes an action:

```bash
python -m server.bio.app restart   # stop whatever is running, then start
python -m server.bio.app stop
python -m server.bio.app status    # which PIDs hold the port, and are they ours
```

`start` is the default, so the bare command above is unchanged.

!!! note "Why not `kill $(lsof -ti:8005)`"
    Auto-reload runs a **reloader parent and a worker sharing one socket**, so `lsof` returns two
    PIDs and killing the worker alone just makes the parent respawn it. `stop` signals the parent
    and waits for the *port* to free, not for a PID to disappear. It also refuses to kill a process
    that is not this service, so an unrelated program on 8005 gets reported rather than terminated.

## Warm the caches before a live session

Every cache is **in memory**. A restart empties them, so warm *after* launching and don't restart
afterwards. Cold vs warm, measured on a laptop:

| | cold | warm |
|---|---|---|
| Base prediction (BRCA1) | 5.7 s | 0.10 s |
| Novel-site candidates (DHX29) | 3.7 s | 0.004 s |
| Gene list for a new annotation | 4.5–54 s | 0.02 s |

```bash
# base predictions
for g in BRCA1 TP53 SERPINA1 TARDBP UNC13A STMN2; do
  curl -s "http://localhost:8005/api/genome/$g/predict?model=openspliceai" -o /dev/null
done

# novel-site candidates
for g in DHX29 ATP6V1A TPR; do
  curl -s "http://localhost:8005/api/novel/$g/candidates?top_k=20" -o /dev/null
done

# base-vs-meta overlay for the showcase genes
python examples/UI_integration/07_warm_ui_cache.py
```

The `threshold` parameter is deliberately absent above: the prediction cache is keyed on
`(gene, model)` only, because raw per-position probabilities don't depend on the cutoff. Only the
cheap classification re-runs when you move a slider, so warming at any threshold warms all of them.

## What needs prebuilt data

Three features read artifacts that must exist on disk first. Everything else works out of the box.

| Feature | Needs | If missing |
|---|---|---|
| **Meta overlay** on the genome view | a per-gene feature `.npz` under `output/meta_layer/ui_cache/gene_cache/` | 404 naming the exact command to build it |
| **Novel Site Explorer** | the M3 eval cache (held-out chromosomes 1/3/5/7/9) | 404 naming the servable universe |
| **Annotation tracks** | `splice_sites_track.parquet` per annotation | tracks degrade silently; the prediction still renders |

Build a feature cache for a new gene:

```bash
python examples/UI_integration/02_build_showcase_feature_cache.py --genes TARDBP
```

That takes ~35 s per gene **provided the conservation bigWigs are cached locally**. They are ~15.8 GB
and only fetched once:

```bash
python examples/UI_integration/02_build_showcase_feature_cache.py --genes TARDBP --download-tracks
```

Without a local copy the extractor streams from UCSC, and a network failure there used to produce a
cache with two silently-zeroed channels. It now refuses to write the gene instead — see
[Reading the numbers](05_reading_the_numbers.md#dead-channels).

Build the annotation track parquets (one-off, ~10 s):

```bash
python examples/data_preparation/05_build_annotation_track_parquets.py
```

## Sanity check

```bash
for p in / /metrics /genome/TARDBP /novel/DHX29; do
  echo "$p -> $(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8005$p")"
done
```

Four 200s means the pages render. It does **not** mean the meta overlay or novel-site data are
present for a given gene — those fail per-gene with a 404 that tells you what to run.

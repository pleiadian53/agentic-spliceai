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

## Two caches, and the order they go in

The Lab has caches at two levels, and they are easy to confuse because both get called "warming".

| | On-disk feature cache | In-memory prediction cache |
|---|---|---|
| Holds | 9 dense multimodal channels per gene, as `.npz` | finished predictions, keyed `(gene, model)` |
| Built by | `02_build_showcase_feature_cache.py` | `07_warm_ui_cache.py`, or any request |
| Cost | ~35 s per gene | ~1–6 s per gene |
| Survives a restart | **yes** | no |

**Disk first, memory second.** Requesting a prediction cannot build a feature cache: the base-only
path never reads a multimodal channel, so no amount of `curl`-ing the base endpoint produces one.
The meta overlay is the only consumer, and it 404s when the `.npz` is absent.

## What needs prebuilt data

Three features read artifacts that must exist on disk first. Everything else works out of the box.

| Feature | Needs | If missing |
|---|---|---|
| **Meta overlay** on the genome view | a per-gene feature `.npz` under `output/meta_layer/ui_cache/gene_cache/` | 404 naming the exact command to build it |
| **Novel Site Explorer** | the M3 eval cache (held-out chromosomes 1/3/5/7/9) | 404 naming the servable universe |
| **Annotation tracks** | `splice_sites_track.parquet` per annotation | tracks degrade silently; the prediction still renders |

The showcase set is already built, so for a standard demo there is nothing to do here. Build a
feature cache when you want the overlay on a **new** gene:

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

## Warm the caches before a live session

This is the in-memory half, and it is per-session: a restart empties it, so warm *after* launching
and don't restart afterwards. Cold vs warm, measured on a laptop:

| | cold | warm |
|---|---|---|
| Base prediction (BRCA1) | 5.7 s | 0.10 s |
| Novel-site candidates (DHX29) | 3.7 s | 0.004 s |
| Gene list for a new annotation | 4.5–54 s | 0.02 s |

```bash
# base predictions (works for any gene — no prebuilt data needed)
for g in BRCA1 TP53 SERPINA1 TARDBP UNC13A STMN2; do
  curl -s "http://localhost:8005/api/genome/$g/predict?model=openspliceai" -o /dev/null
done

# novel-site candidates
for g in DHX29 ATP6V1A TPR; do
  curl -s "http://localhost:8005/api/novel/$g/candidates?top_k=20" -o /dev/null
done

# base-vs-meta overlay — showcase genes only, since each needs its .npz
python examples/UI_integration/07_warm_ui_cache.py
```

SERPINA1 appears in the base loop but not in the overlay set: it has no feature cache, so it is a
base-only demo gene. `07_warm_ui_cache.py` defaults to the eight genes `02_` builds, and the two
lists are kept in sync deliberately, so the bare command is the one you want for a demo.

To warm a different set, `--genes` takes several names, space- or comma-separated:

```bash
python examples/UI_integration/07_warm_ui_cache.py --genes TP53 SOD1
python examples/UI_integration/07_warm_ui_cache.py --genes TP53,SOD1   # same thing
```

If it reports failures, read *which* request failed. A **base** failure is the server or the gene
symbol, since the base path reads no prebuilt data at all. A **meta overlay** failure is the missing
`.npz`, and the script prints the exact `02_` command to build it.

The `threshold` parameter is deliberately absent above: the prediction cache is keyed on
`(gene, model)` only, because raw per-position probabilities don't depend on the cutoff. Only the
cheap classification re-runs when you move a slider, so warming at any threshold warms all of them.

### Checking that the warm-up stuck

```bash
curl -s http://localhost:8005/api/debug/cache | jq
```

Reports each prediction cache's occupancy against its capacity, its entries **oldest-first**, and
which models are loaded:

```
base                   2/50   next_evicted: TP53
meta overlay           1/50   next_evicted: TARDBP
novel candidates (M3)  0/50
models loaded          base: openspliceai   meta: m2s.concat_fusion.cleanannot
```

Two things this answers that the logs don't:

- **Whether warming survived.** All three caches share one capacity
  (`config.MAX_CACHED_PREDICTIONS`), so browsing during a session can evict a gene warmed for it.
  `next_evicted` names the entry that goes first.
- **Whether the *model* is loaded**, which is the other half of latency. A cache miss on a loaded
  model costs a second or two; a miss on an unloaded one costs a model load, and for SpliceAI that
  is five TensorFlow models.

!!! warning "Dev-only"
    `/api/debug/*` exposes internal server state. It returns gene symbols and model names only, with
    no filesystem paths and no request history, but it is server internals rather than product
    surface. It defaults **on**, because the question it answers is needed exactly when remembering
    a flag is least likely. Disable with `BIO_LAB_DEBUG=0`, which unregisters the route entirely so
    it 404s indistinguishably from any unknown path.

## Sanity check

```bash
for p in / /metrics /genome/TARDBP /novel/DHX29; do
  echo "$p -> $(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8005$p")"
done
```

Four 200s means the pages render. It does **not** mean the meta overlay or novel-site data are
present for a given gene — those fail per-gene with a 404 that tells you what to run.

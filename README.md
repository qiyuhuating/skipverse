# ✦ skipverse

**A vector database written from scratch — HNSW graph index, crash-safe WAL, deterministic replay, and a living visualization of every greedy hop.**

[![CI](https://github.com/qiyuhuating/skipverse/actions/workflows/ci.yml/badge.svg)](https://github.com/qiyuhuating/skipverse/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-live-38bdf8)](https://qiyuhuating.github.io/skipverse/)
[![zero dependencies](https://img.shields.io/badge/runtime%20deps-0-34d399)](#zero-dependencies)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

| the graph | the search | the answer |
|---|---|---|
| ![](docs/screenshot-graph.png) | ![](docs/screenshot-search.png) | ![](docs/screenshot-results.png) |

*The demo is the same engine that ships on npm — running in your browser. Watch the green token dive through layers 2 → 1 → 0, then the gold rings light up the top-k. [Live demo →](https://qiyuhuating.github.io/skipverse/)*

![](docs/demo.gif)

---

## Why from scratch

Vector search powers every embedding-based system, yet the core algorithm —
[HNSW](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2016) — is usually swallowed by a dependency. skipverse
implements the whole stack by hand, in ~1,450 lines of strict TypeScript with **zero runtime dependencies**:

- **multi-layer navigable small world graph** — exponential layer assignment, greedy descent through sparse upper
  layers, ef-bounded beam search on the dense layer 0;
- **diversity-aware neighbor selection** (paper's Algorithm 4, `keepPruned=false`) so hubs don't eat the graph;
- **durability the honest way** — CRC-32 framed WAL, torn-write recovery, atomic snapshot checkpoints, op-sequence
  replay;
- **determinism** — layer assignment is seeded from a hash of the vector id plus the insertion counter, so replaying
  the same op stream rebuilds a byte-identical graph (and the tests rely on it);
- **one isomorphic core** — the same `HnswIndex` class serves HTTP traffic on Node and renders this README's demo in
  the browser.

## Quickstart

### Embed it

```ts
import { HnswIndex } from "skipverse";

const idx = new HnswIndex({ dim: 64, metric: "cosine", M: 16, efConstruction: 200 });
idx.add("doc-1", embedding);
idx.add("doc-2", embedding);

const hits = idx.search(query, 10, { ef: 64 });
// [{ id: "doc-42", dist: 0.113 }, ...]
```

### Persist it

```ts
import { VectorStore } from "skipverse";

const store = VectorStore.open({ dataDir: "./data", dim: 64, metric: "cosine", quantization: "sq8" });
store.upsert("doc-1", embedding);        // appended to CRC-framed WAL, auto-checkpointed
store.calibrate();                       // freeze ranges + rewrite codes + rotate snapshot
store.remove("doc-2");                   // soft delete: filtered from results, kept as graph anchor
store.compact();                         // rebuild from alive vectors, reclaim tombstone space
// kill -9 at any point: reopen replays the WAL, torn tails are truncated at frame boundaries
```

### Serve it

```bash
git clone https://github.com/qiyuhuating/skipverse && cd skipverse
npm install && npm run build
npx skipverse serve --port 8787 --dim 64 --metric cosine
```

```bash
curl -X POST localhost:8787/vectors -H 'content-type: application/json' \
     -d '{"vectors":[{"id":"a","vec":[0.1,0.2, ...]}]}'

curl -X POST localhost:8787/search \
     -d '{"vec":[0.1,0.2, ...],"k":8,"ef":64,"trace":true}'
```

### Watch it

```bash
git clone https://github.com/qiyuhuating/skipverse && cd skipverse
npm install && npm run serve     # → http://localhost:8787
```

## Filter, compact, quantize

**Filtered search** — hnswlib semantics: filtered-out nodes are still traversed (the graph stays connected) but never
occupy the result beam, so the whole `ef` budget is spent on admissible nodes. Raise `ef` when the filter is very
selective.

```ts
const hits = idx.search(q, 10, { ef: 128, filter: (id) => tenantOf(id) === "acme" });
```

**Compaction** — deletes are tombstones: dead nodes stay as traversal anchors until you reclaim them.

```ts
store.compact();   // deterministic rebuild from alive vectors + atomic snapshot rotation
```

**SQ8 scalar quantization** — the faiss/hnswlib "train once, serve" model: insert in f32, calibrate once, every
stored vector becomes one byte per dimension (4× smaller); distances then run in register-dequantized space.

```ts
const idx = new HnswIndex({ dim: 64, metric: "euclidean", quantization: "sq8" });
// ... add() your corpus in f32 ...
idx.calibrate();     // freezes per-dimension [min,max], rewrites vectors as codes — one-shot
idx.add("new-doc", vec);   // later inserts are quantized through the frozen ranges
```

`quantization: "sq4"` packs the same affine map into 4-bit nibbles (8× smaller, noticeably lossy — see the honest
benchmark below).

`quantization: "pq"` (product quantization) goes further: 32 subspaces each get their own deterministic k-means
codebook (256 centroids), a vector becomes one codebook index per subspace — 8 B/vec for 64d, **32× smaller**. At
equal 8× compression PQ scores **0.742 recall@10 where SQ4 scores 0.589** on the benchmark below; the price is
k-means training time during `calibrate()` (~3× the SQ build). Distances run through a per-query m×256 asymmetric
distance-computation table — one table build per query, then one lookup per subspace per candidate.

Under the hood the quantized kernels do real math, not byte-tricks. For **euclidean**, search uses *asymmetric
distance computation*: the query stays exact f32 (reduced to `v⊙step`, `v·min`, `‖v‖²`) and each pair costs one
dim-loop plus three scalars against the dequantized code. For **cosine/dot**, both sides are u8 codes evaluated via
the identity `v·v′ = q·W·q′ + c·q + c·q′ + K` (with `W = step²`, `c = step·min`, `K = Σmin²`) — one weighted dot plus
three precomputed scalars. The per-node aux scalars are pure functions of (codes, calibration), so they're rebuilt
on load instead of stored. Serialized as format v2 (u8 flag + calibration table); v1 indexes still load.

## The trace: search you can read

`trace: true` returns the full traversal, layer by layer — this is what the demo animates:

```json
{
  "results": [{ "id": "564", "dist": 0.089 }, ...],
  "trace": {
    "layers": [
      { "level": 2, "entry": "712", "hops": [{ "from": "712", "to": "709", "dist": 0.62 }], "visited": 2 },
      { "level": 1, "entry": "709", "hops": [ ... 11 more ... ], "visited": 12 },
      { "level": 0, "entry": "709", "hops": [ ... 139 hops ... ], "visited": 141 }
    ],
    "visitedTotal": 155
  }
}
```

On the 10k×64d benchmark, answering a query touches **~500 of 10,000 nodes** — the other 9,500 are never looked at.

## How it works

```
            entry                                        query
              │                                             │
   layer 2    │        7─────────9                          │  1. greedy descent: walk the sparse
              │      /           \                          │     upper layers toward the query's
   layer 1    │  48───────48───────48──                     │     neighborhood          O(log N)
              │  (beam widens: ef grows per layer)          │
   layer 0    │  800 nodes · degree ≤ 2M                    │  2. beam search ef nearest on the
              │  ~~~~~~~~~~~~~~~~~~~~~~~~~~~                │     dense bottom layer    O(M · ef)
              └─ every node lives on all layers ≤ its own   │  3. diversity-heuristic keeps links
                                                               spread out during insertion
```

**Insert** — sample the node's top layer from an exponential distribution, greedy-descend to it, then per layer:
ef-bounded search for candidates, Algorithm-4 selection of M diverse links, bidirectional connect, and re-select any
neighbor that overflowed its degree cap (M on upper layers, 2M on layer 0).

**Search** — greedy from the entry point down to layer 1, then a `max(ef, k)`-bounded beam on layer 0. `ef` is the
recall/latency knob: benchmark below.

**Delete** — soft: nodes stay as traversal anchors, are filtered from results, and vanish at the next snapshot
checkpoint. Hard deletes in a proximity graph are a research topic; this is the hnswlib-grade pragmatic answer, and
the store tracks `deleted` so compaction is observable.

**Durability** —

```
upsert ──► WAL frame [len | crc32 | seq | op | id | f32×dim] ──► fsync-less append (batch durability)
                │ crash mid-frame?
                ▼
reopen: CRC scan ──► longest valid prefix ──► truncate torn tail ──► replay ops with seq > snapshot.lastSeq
checkpoint: snapshot.tmp ──► atomic rename ──► WAL truncate
compact:   deterministic rebuild from alive vectors ──► tombstone space reclaimed
```

## Benchmarks

Deterministic, seeded, reproducible (`npm run bench`):

### skipverse 10,000 × 64d · M=16 · efConstruction=200 · euclidean · k=10

| mode | efSearch | recall@10 | QPS | avg nodes visited |
|:-----|---------:|----------:|----:|------------------:|
| f32 | 16 | 0.9650 | 10,911 | 288.9 |
| f32 | 64 | 1.0000 | 4,952 | 501.9 |
| f32 | 128 | 1.0000 | 3,467 | 571.1 |
| sq8 (ADC) | 16 | 0.9400 | 13,111 | 288.1 |
| sq8 (ADC) | 64 | 0.9720 | 6,701 | 502.1 |
| sq8 (ADC) | 128 | 0.9720 | 4,169 | 570.9 |
| sq4 (ADC) | 16 | 0.5990 | 6,921 | 296.5 |
| sq4 (ADC) | 64 | 0.5890 | 3,273 | 504.6 |
| sq4 (ADC) | 128 | 0.5880 | 2,566 | 574.0 |
| pq (ADC) | 16 | 0.7330 | 4,686 | 289.7 |
| pq (ADC) | 64 | 0.7420 | 2,893 | 503.7 |
| pq (ADC) | 128 | 0.7420 | 2,243 | 573.3 |
| brute force | — | 1.0000 | 197 | 10,000 |

build: f32 2.70s · sq8 2.86s · sq4 4.06s · pq 12.16s (k-means training) · vector storage: 256 → 64 → 32 → **8 B/vec** (4× / 8× / 32×).

Reading the table honestly: at ef=64 the f32 index is **~17× brute force at identical recall** (46× at ef=16).
Queries are in-distribution (data point + N(0, 0.3) noise) — deliberately off-manifold queries are the known weak
spot of greedy graph search, in this implementation and every other.

**SQ8 is the sweet spot**: 4× smaller vectors at ~117% of the f32 QPS and −2.8 recall points at ef=64. The win comes
from asymmetric distance computation — the query stays full-precision f32 and only the stored side is a u8 code,
which removes half the quantization noise (symmetric codes scored 0.864 on this dataset before ADC; the residual
plateau at 0.972 is the data-side noise).

**SQ4 is the extreme tier**: 8× smaller at ~0.59 recall on this dataset — 16 levels per dimension is inherently
lossy, and this clustered geometry is full of near-tie distances that 4-bit codes cannot separate (spread-out
distributions lose far less). Treat it as a coarse-filter stage or a memory-last resort; rescoring with original
vectors is the natural next layer and lives on the roadmap.

## API reference

### HnswIndex (embeddable, isomorphic)

| signature | what it does | notes |
|:--|:--|:--|
| `new HnswIndex({ dim, metric?, M?, efConstruction?, seed?, quantization?, pqSubspaces?, extendCandidates? })` | create an index | `metric`: cosine (default) / euclidean / dot · `M` 16 · `efConstruction` 200 · `seed` pins layer assignment · `quantization`: none / sq8 / sq4 / pq |
| `add(id, vec)` | insert or replace | replace soft-deletes the old vector · vec: `number[]` / `Float32Array` |
| `remove(id)` | soft delete | stays as traversal anchor, filtered from results · returns `false` if absent |
| `search(q, k?, { ef?, filter? })` | k nearest | `ef` defaults to `max(k, 16)` · `filter(id)`: excluded nodes are traversed, never returned |
| `searchWithTrace(q, k?, opts?)` | search + traversal | returns `{ results, trace }` — the demo animates `trace.layers` |
| `vector(id)` | stored vector | dequantized after calibration · null when absent/deleted |
| `adjacency(id)` | per-level neighbor ids | for visualization / introspection |
| `compacted()` | rebuild without tombstones | deterministic · calibration carries over |
| `calibrate()` | freeze ranges, rewrite codes | one-shot, requires `quantization: "sq4" | "sq8"` |
| `serialize()` / `HnswIndex.deserialize(data)` | binary round-trip | format v2 · v1 indexes still load |
| `stats()` | counts, per-level degree stats | |
| `size` · `deletedCount` · `isCalibrated` · `bytesPerVector` | live metrics | |

### SearchPool (concurrent reads, Node)

| signature | what it does | notes |
|:--|:--|:--|
| `new SearchPool({ store, workers? })` | N workers hold read-only snapshots | default: `cpus−1` capped at 4 |
| `searchAsync(vec, k?, ef?)` | search off the main thread | promise per request, round-robin dispatch |
| `refresh()` | broadcast a fresh snapshot | required after writes — snapshots are explicit |
| `close()` | drain and exit all workers | in-flight searches settle first |

Single file, worker re-enters the module itself, zero dependencies. Results are identical to the synchronous path.

### VectorStore (durable, Node)

| signature | what it does | notes |
|:--|:--|:--|
| `VectorStore.open({ dataDir, dim, metric?, M?, efConstruction?, seed?, quantization?, fsync?, checkpointEvery? })` | open or create | validates meta.json on reopen · `fsync: true` = real power-loss durability |
| `upsert(id, vec)` / `upsertBatch(entries)` | writes | batch = one WAL syscall (one fsync per batch) |
| `remove(id)` · `get(id)` | tombstone / read back | `get` is dequantized post-calibration |
| `search(q, k?, opts?)` / `searchWithTrace(...)` | queries | delegates to the index |
| `calibrate()` | calibrate + rotate snapshot | |
| `compact()` | rebuild + snapshot rotation | returns `{ before, after }` |
| `checkpoint()` | snapshot + WAL truncate | automatic every `checkpointEvery` ops (default 4096) |
| `close(checkpoint?)` | release lock + fd | |
| `info()` | stats + WAL bytes | |

Single-writer: a store held by a live process refuses a second `open`; stale locks from dead processes are cleared automatically.

### HTTP API

| route | body | response |
|:--|:--|:--|
| `POST /vectors` | `{ vectors: [{id, vec}] }` | `{ upserted, count }` |
| `GET /vectors/:id` | — | `{ id, vec }` · 404 if absent |
| `DELETE /vectors/:id` | — | `{ removed }` |
| `POST /search` | `{ vec, k?, ef?, trace? }` | `{ results, tookMs }` · + `trace` when asked |
| `GET /stats` | — | index + WAL stats |
| `POST /checkpoint` | — | `{ ok }` |
| `GET /healthz` | — | `{ ok, count }` |

Every response carries an `x-response-time` header.

### CLI

| command | flags | purpose |
|:--|:--|:--|
| `skipverse serve` | `--port --data --dim --metric --M --ef-construction --quantization` | HTTP + demo |
| `skipverse import` | `--file vectors.jsonl` + store flags | bulk load (500/batch) |
| `skipverse calibrate` | store flags | freeze ranges + rewrite codes |

## Zero dependencies

`package.json` lists four **dev**Dependencies (typescript, tsx, esbuild, @types/node) and nothing else. The engine
uses `Math`, `DataView`, `Uint8Array`, and `node:fs` when on Node. That is deliberate: a database's trust boundary
should not include 200 transitive packages, and the algorithm is the product.

## Acceptance criteria (enforced by CI)

- 77 tests green on Node 22 & 24 — including recall floors for f32/sq8/sq4, degree-cap invariants, filtered-search
  semantics, WAL torn-write recovery, snapshot+WAL round-trips, compaction, a full HTTP restart cycle, and a
  2,500-op seeded fuzz run where a brute-force mirror model must agree with the store at every step;
- typecheck strict, `verbatimModuleSyntax`, no `any` in the engine;
- the CI benchmark run must hold **f32 ≥ 0.95, sq8 ≥ 0.90, sq4 ≥ 0.50, pq ≥ 0.80 recall@10 at ef=64** or the build fails.

## Roadmap

- [ ] SQ4/PQ residual rescoring stage (recover low-bit recall with a second pass)
- [ ] online recalibration when the data distribution drifts
- [x] concurrent readers (worker-thread snapshot pool) — v0.3.3
- [x] product quantization with deterministic k-means training — v0.4.0
- [ ] memory-mapped snapshot loader (zero-copy warm start)
- [ ] WASM build of the core for CDN-drop usage

## License

[MIT](LICENSE) © 2026 qiyuhuating

# Changelog

## 0.4.0 — 2026-10-06

- **Product quantization (PQ)**: quantization: "pq" with pqSubspaces — deterministic k-means codebooks (k-means++ + 12 Lloyd iterations on mulberry32), asymmetric ADC via a per-query m×256 table. **At the same 8× compression PQ scores 0.742 recall@10 where SQ4 scores 0.589** (+15 points); trade is ~3× slower calibrate (k-means training). Version-3 serialized; older indexes still load.
- **Concurrent read pool**: SearchPool — worker threads hold read-only snapshots; searches never block the writer.
- **Fixed: single-id upsert chains disconnected the graph** (found by the invariant suite) — empty-beam inserts now anchor to recent nodes.
- **Fixed: calibrated euclidean inter-node distances** used step instead of step² (adversarial review, reproduced).
- **Fixed: add(rawCodeUint8Array) crashed** on calibrated euclidean indexes (missing ADC anchor).
- **extendCandidates** implemented and honestly benchmarked: +0.006 recall for ~2× build — kept, default off.
- **Invariant suite** (11 tests) + adversarial-review fixes (compacted carries extendCandidates; M ≤ 32767).
- README API reference (HnswIndex / SearchPool / VectorStore / HTTP / CLI). 77 tests green.

## 0.3.3 — 2026-10-06

- **Concurrent read pool**: `SearchPool` (src/node/pool.ts) spawns worker threads holding read-only index
  snapshots — a search storm never blocks the main-thread writer and vice versa. Self-entry worker (single
  zero-dependency file), round-robin dispatch, explicit `refresh()` snapshot semantics, graceful close. Results are
  identical to the synchronous path (tests assert exact equality).
- **extendCandidates option** (paper Algorithm 4): implemented and benchmarked honestly — +0.006 recall on
  off-manifold queries for ~2× build cost, zero gain in-distribution. Rejected as a default (the diversity
  heuristic already covers it); kept as a build-time knob with a default-off equivalence test.
- README: full API reference tables for HnswIndex / SearchPool / VectorStore / HTTP / CLI.
- **Invariant test suite**: WAL mid-stream corruption boundaries, compaction idempotence, cross-metric ef
  monotonicity, unicode/oversized ids, zero-vector safety, upsert-chain accounting, WAL replay determinism (11 new
  tests).

## 0.3.2 — 2026-10-06

- **Batched WAL writes**: `upsertBatch` validates all vectors, applies them, and flushes the whole batch in a single
  write syscall — under `fsync: true` that means one fsync per batch instead of one per vector.
- `skipverse serve` now shuts down gracefully on SIGINT (lock released, no stale-lock warning on next start).
- CI runs the quickstart example as a gate.

## 0.3.1 — 2026-10-06

- **2× hot-path optimization**: searchLayer now uses a numeric (dist, handle) binary heap with zero per-element
  allocation plus a generation-stamped visited array instead of a `Set` — build 2× faster (3.4s → 1.7s for 10k×64d),
  search QPS up 1.6-2.3× across all quantization modes, recall bit-identical.
- **WAL fsync option** (`fsync: true`): real power-loss durability for appends and checkpoints (Windows needs a
  write handle for FlushFileBuffers — the test caught that).
- **Single-writer lockfile**: opening a store held by a live process throws; stale locks from dead processes are
  auto-cleared.
- **Introspection API**: `HnswIndex.vector(id)` / `store.get(id)` (dequantized post-calibration); HTTP `GET
  /vectors/:id`, `x-response-time` header and `tookMs` on search responses.
- `examples/quickstart.ts` — the 60-second tour (`npm run example`).

## 0.3.0 — 2026-10-06

- **SQ4 scalar quantization**: `quantization: "sq4"` packs 4-bit codes (two dimensions per byte, 8× smaller than
  f32) behind the same one-shot `calibrate()` flow. All three metric kernels are nibble-aware; euclidean keeps ADC.
  Benchmarked honestly: 8× memory at ~0.59 recall@10 on tight 64d clusters — coarse-stage material.
- **Quantization goes full-stack**: `VectorStore` accepts `quantization`, persists it in meta.json (with conflict
  validation), and gains `store.calibrate()`; the CLI grows `--quantization sq8|sq4` and a `skipverse calibrate`
  subcommand.
- **Seeded fuzz test**: 2,500 mixed upsert/delete/search/checkpoint ops against a brute-force mirror model that must
  agree after every search, plus randomized serialize/deserialize churn across metrics and dims.
- CI benchmark gates extended: f32 ≥ 0.95, sq8 ≥ 0.90, sq4 ≥ 0.50 recall@10 @ef64.

## 0.2.0 — 2026-10-04

- **SQ8 scalar quantization** with one-shot `calibrate()`: per-dimension affine u8 codes, 4× smaller vectors.
  Euclidean search uses **asymmetric distance computation** (exact f32 query × dequantized code): recall@10 @ef64
  0.972 vs f32's 1.000 at 4× less memory. Cosine/dot run a symmetric expansion kernel
  (`v·v′ = q·W·q′ + c·q + c·q′ + K`). Fixes found by the new tests: the aux projection must be the code-space
  `c·q` (not `c·v`), `dot` metric builds aux, deserialize builds the ADC kernel.
- **Filtered search**: `search(q, k, { filter })` — filtered nodes are traversed but never enter the result beam
  (hnswlib semantics); soft-deleted nodes behave identically.
- **Compaction**: `HnswIndex.compacted()` / `VectorStore.compact()` — deterministic rebuild from alive vectors,
  snapshot rotation reclaims tombstones.
- `docs/demo.webm` recorded from the live demo; README demo GIF.
- Deserialization hardened: aux scalars are pure functions of (codes, calibration) and are rebuilt on load.

## 0.1.0 — 2026-10-02

- HNSW approximate-nearest-neighbor index from scratch (Algorithm 4 neighbor selection, ef-bounded beam search,
  soft deletes, deterministic layer assignment).
- CRC-32 framed WAL with torn-write recovery, atomic snapshot checkpoints, seq-based replay.
- HTTP API + CLI (`serve`, `import`) + embeddable library API; traversal trace for visualization.
- Living search visualization (canvas) served at `/` and on GitHub Pages.
- 28 tests, reproducible benchmarks, zero runtime dependencies.

# Changelog

## 0.3.0 — 2026-10-04

- **SQ4 scalar quantization**: `quantization: "sq4"` packs 4-bit codes (two dimensions per byte, 8× smaller than
  f32) behind the same one-shot `calibrate()` flow. All three metric kernels are nibble-aware; euclidean keeps ADC.
  Benchmarked honestly: 8× memory at ~0.59 recall@10 on tight 64d clusters — coarse-stage material.
- **Seeded fuzz test**: 2,500 mixed upsert/delete/search/checkpoint ops against a brute-force mirror model that must
  agree after every search, plus randomized serialize/deserialize churn across metrics and dims. Found (and this
  release fixes) tombstone-accounting assumptions in the test itself — the store semantics are now pinned.
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

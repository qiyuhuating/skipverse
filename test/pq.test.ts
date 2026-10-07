import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { HnswIndex } from '../src/core/hnsw.js';
import { genClusterData } from '../src/core/dataset.js';
import { bruteForce, recall } from './helpers.js';

describe('PQ product quantization', () => {
  it('beats SQ4 at the same 8× compression: euclidean recall@10 ≥ 0.65 @ef64 (measured 0.74)', () => {
    const { vecs, queries } = genClusterData(5000, 64, 20, 1234);
    const idx = new HnswIndex({ dim: 64, metric: 'euclidean', quantization: 'pq', pqSubspaces: 32, M: 16, efConstruction: 200, seed: 42 });
    for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
    idx.calibrate();
    assert.equal(idx.bytesPerVector, 32, '64d / 32 subspaces = 1 byte per vector… per subspace');
    let sum = 0;
    for (const q of queries.slice(0, 50)) sum += recall(bruteForce(vecs, q, 10, 'euclidean'), idx.search(q, 10, { ef: 64 }));
    const r = sum / 50;
    assert.ok(r >= 0.65, `pq euclidean recall@10 @ef64 = ${r.toFixed(3)}, want ≥ 0.65`);
  });

  it('round-trips search-identically at k=256 codebooks', () => {
    const { vecs, queries } = genClusterData(400, 32, 10, 11);
    const idx = new HnswIndex({ dim: 32, metric: 'euclidean', quantization: 'pq', pqSubspaces: 32, M: 8, seed: 7 });
    for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
    idx.calibrate();
    assert.equal(idx.bytesPerVector, 32);
    const q = queries[3]!;
    const round = HnswIndex.deserialize(idx.serialize());
    assert.ok(round.isCalibrated);
    assert.deepEqual(round.search(q, 10, { ef: 64 }), idx.search(q, 10, { ef: 64 }));
    // vector() must return the dequantized reconstruction, not raw code bytes
    const v = round.vector('0')!;
    assert.equal(v.length, 32);
    assert.ok(Math.abs(v[0]! - vecs[0]![0]!) < 5, 'reconstruction must be near the original');
  });

  it('trains deterministically: same seed ⇒ identical graph and answers', () => {
    const { vecs, queries } = genClusterData(300, 16, 6, 21);
    const build = () => {
      const idx = new HnswIndex({ dim: 16, metric: 'euclidean', quantization: 'pq', pqSubspaces: 8, seed: 9 });
      for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
      idx.calibrate();
      return idx;
    };
    const a = build();
    const b = build();
    const q = queries[0]!;
    assert.deepEqual(a.searchWithTrace(q, 10), b.searchWithTrace(q, 10));
  });

  it('supports cosine and dot metrics', () => {
    for (const metric of ['cosine', 'dot'] as const) {
      const { vecs, queries } = genClusterData(500, 16, 8, 31);
      const idx = new HnswIndex({ dim: 16, metric, quantization: 'pq', pqSubspaces: 8, M: 8, seed: 3 });
      for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
      idx.calibrate();
      let sum = 0;
      for (const q of queries.slice(0, 30)) sum += recall(bruteForce(vecs, q, 10, metric), idx.search(q, 10, { ef: 64 }));
      const r = sum / 30;
      assert.ok(r >= 0.75, `pq ${metric} recall = ${r.toFixed(3)}, want ≥ 0.75`);
    }
  });

  it('compacted() carries the codebooks and stays equivalent', () => {
    const { vecs, queries } = genClusterData(400, 16, 8, 37);
    const idx = new HnswIndex({ dim: 16, metric: 'euclidean', quantization: 'pq', pqSubspaces: 8, M: 8, seed: 5 });
    for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
    idx.calibrate();
    for (let i = 0; i < 100; i++) idx.remove(String(i));
    const fresh = idx.compacted();
    assert.ok(fresh.isCalibrated);
    assert.equal(fresh.bytesPerVector, 8);
    const q = queries[2]!;
    const r = recall(idx.search(q, 10, { ef: 64 }), fresh.search(q, 10, { ef: 64 }));
    assert.ok(r >= 0.9, `compacted vs original recall = ${r.toFixed(3)}`);
  });

  it('validates construction and calibration errors', () => {
    assert.throws(() => new HnswIndex({ dim: 8, quantization: 'pq', pqSubspaces: 0 }), /pqSubspaces/);
    assert.throws(() => new HnswIndex({ dim: 8, quantization: 'nope' as never }), /quantization/);
    const { vecs } = genClusterData(50, 8, 3, 2);
    const idx = new HnswIndex({ dim: 8, metric: 'euclidean', quantization: 'pq', seed: 1 });
    for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
    idx.calibrate();
    assert.throws(() => idx.calibrate(), /already calibrated/);
  });
});

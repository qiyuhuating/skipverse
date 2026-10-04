import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { HnswIndex } from '../src/core/hnsw.js';
import { genClusterData } from '../src/core/dataset.js';
import { bruteForce, recall } from './helpers.js';

function buildSq8(n: number, dim: number, clusters: number, seed: number, metric: 'euclidean' | 'cosine', calibrate = true) {
  const { vecs, queries } = genClusterData(n, dim, clusters, seed);
  const idx = new HnswIndex({ dim, metric, quantization: 'sq8', M: 12, efConstruction: 150, seed: 5 });
  for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
  if (calibrate) idx.calibrate();
  return { idx, vecs, queries };
}

describe('SQ8 scalar quantization', () => {
  it('keeps euclidean recall ≥ 0.85 after calibration', () => {
    const { idx, vecs, queries } = buildSq8(3000, 16, 25, 7, 'euclidean');
    assert.ok(idx.isCalibrated);
    let sum = 0;
    for (const q of queries.slice(0, 50)) {
      sum += recall(bruteForce(vecs, q, 10, 'euclidean'), idx.search(q, 10, { ef: 64 }));
    }
    const r = sum / 50;
    assert.ok(r >= 0.85, `sq8 euclidean recall@10 @ef64 = ${r.toFixed(3)}, want ≥ 0.85`);
  });

  it('keeps cosine recall ≥ 0.85 after calibration', () => {
    const { idx, vecs, queries } = buildSq8(3000, 16, 25, 7, 'cosine');
    let sum = 0;
    for (const q of queries.slice(0, 50)) {
      sum += recall(bruteForce(vecs, q, 10, 'cosine'), idx.search(q, 10, { ef: 64 }));
    }
    const r = sum / 50;
    assert.ok(r >= 0.85, `sq8 cosine recall@10 @ef64 = ${r.toFixed(3)}, want ≥ 0.85`);
  });

  it('shrinks vector storage 4× (links dominate the file, vectors do not)', () => {
    const { vecs } = genClusterData(600, 32, 8, 11);
    const plain = new HnswIndex({ dim: 32, metric: 'euclidean', quantization: 'sq8', seed: 1 });
    const quant = new HnswIndex({ dim: 32, metric: 'euclidean', quantization: 'sq8', seed: 1 });
    for (let i = 0; i < vecs.length; i++) {
      plain.add(String(i), vecs[i]!);
      quant.add(String(i), vecs[i]!);
    }
    assert.equal(plain.bytesPerVector, 128, 'uncalibrated sq8 stores f32');
    assert.equal(quant.bytesPerVector, 128);
    quant.calibrate();
    assert.equal(quant.bytesPerVector, 32);
    const before = plain.serialize().byteLength;
    const after = quant.serialize().byteLength;
    // total file also contains ids + graph links, so expect < 60%, not < 25%
    assert.ok(after < before * 0.6, `serialized ${after} should be < 60% of ${before}`);
    // per-vector payload is the honest 4×
    assert.equal(quant.bytesPerVector / plain.bytesPerVector, 0.25);
  });

  it('refuses to calibrate twice, empty, or outside sq8 mode', () => {
    const { idx } = buildSq8(50, 8, 3, 2, 'euclidean', false);
    assert.ok(!idx.isCalibrated);
    assert.throws(() => new HnswIndex({ dim: 8, metric: 'euclidean' }).calibrate(), /quantization/);
    assert.throws(() => new HnswIndex({ dim: 8, metric: 'euclidean', quantization: 'sq8' }).calibrate(), /empty/);
    idx.calibrate();
    assert.throws(() => idx.calibrate(), /already calibrated/);
  });

  it('round-trips calibrated indexes (ids exact, distances within f32 aux error)', () => {
    const { idx, vecs, queries } = buildSq8(500, 16, 10, 13, 'cosine');
    const q = queries[3]!;
    const before = idx.search(q, 10, { ef: 64 });
    const restored = HnswIndex.deserialize(idx.serialize());
    assert.ok(restored.isCalibrated);
    // aux scalars are stored as f32, so distances agree to ~1e-6 but not bit-exact
    const after = restored.search(q, 10, { ef: 64 });
    assert.equal(after.length, before.length);
    before.forEach((r, i) => {
      assert.equal(r.id, after[i]!.id);
      assert.ok(Math.abs(r.dist - after[i]!.dist) < 1e-5, `dist ${r.dist} vs ${after[i]!.dist}`);
    });
    assert.deepEqual(restored.stats(), idx.stats());
    void vecs;
  });

  it('accepts inserts after calibration through the frozen ranges', () => {
    const { idx, vecs } = buildSq8(800, 16, 10, 17, 'euclidean');
    for (let i = 800; i < 900; i++) idx.add(String(i), vecs[i % 800]!);
    assert.equal(idx.size, 900);
    const got = idx.search(vecs[850 % 800]!, 5, { ef: 64 });
    assert.ok(got.length === 5);
    assert.ok(got.some((r) => Number(r.id) >= 800), 'post-calibration inserts must be reachable');
  });

  it('stays deterministic across replays', () => {
    const { vecs, queries } = genClusterData(300, 8, 5, 19);
    const a = new HnswIndex({ dim: 8, metric: 'euclidean', quantization: 'sq8', seed: 4 });
    const b = new HnswIndex({ dim: 8, metric: 'euclidean', quantization: 'sq8', seed: 4 });
    for (let i = 0; i < vecs.length; i++) {
      a.add(String(i), vecs[i]!);
      b.add(String(i), vecs[i]!);
    }
    a.calibrate();
    b.calibrate();
    const q = queries[0]!;
    assert.deepEqual(a.searchWithTrace(q, 10), b.searchWithTrace(q, 10));
  });

  it('compacted() carries the calibration over', () => {
    const { idx, queries } = buildSq8(400, 16, 8, 23, 'euclidean');
    for (let i = 0; i < 100; i++) idx.remove(String(i));
    const fresh = idx.compacted();
    assert.ok(fresh.isCalibrated);
    assert.equal(fresh.size, 300);
    assert.equal(fresh.bytesPerVector, 16);
    const q = queries[2]!;
    const r = recall(idx.search(q, 10, { ef: 64 }), fresh.search(q, 10, { ef: 64 }));
    assert.ok(r >= 0.9, `compacted-vs-original recall = ${r.toFixed(3)}`);
  });

  it('upsert and remove keep working on a calibrated index', () => {
    const { idx } = buildSq8(200, 12, 6, 29, 'euclidean');
    assert.ok(idx.remove('5'));
    assert.ok(!idx.remove('5'));
    idx.add('5', [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // far outlier, clamped through frozen ranges
    assert.equal(idx.size, 200);
    assert.ok(idx.search([1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, { ef: 32 }).length > 0);
  });
});

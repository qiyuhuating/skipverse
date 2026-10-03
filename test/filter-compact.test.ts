import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { HnswIndex } from '../src/core/hnsw.js';
import { genClusterData } from '../src/core/dataset.js';
import { bruteForce, recall } from './helpers.js';

describe('filtered search', () => {
  const { vecs, queries } = genClusterData(800, 12, 10, 17);
  const idx = new HnswIndex({ dim: 12, metric: 'euclidean', M: 12, efConstruction: 150, seed: 3 });
  for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
  const q = queries[0]!;

  it('returns only ids passing the predicate, at high recall', () => {
    const filter = (id: string) => Number(id) % 2 === 0;
    const got = idx.search(q, 10, { ef: 96, filter });
    assert.ok(got.length === 10);
    assert.ok(got.every((r) => Number(r.id) % 2 === 0), 'filtered-out ids must not surface');
    const truth = bruteForce(vecs, q, 10, 'euclidean').filter((r) => filter(r.id));
    const r = recall(truth, got);
    assert.ok(r >= 0.9, `filtered recall@10 = ${r.toFixed(3)}, want ≥ 0.90`);
  });

  it('still finds k results when the filter keeps only 10% of the data', () => {
    const filter = (id: string) => Number(id) % 10 === 0;
    const got = idx.search(q, 8, { ef: 128, filter });
    assert.equal(got.length, 8);
    assert.ok(got.every((r) => Number(r.id) % 10 === 0));
    const truth = bruteForce(vecs, q, 8, 'euclidean').filter((r) => filter(r.id));
    const r = recall(truth, got);
    assert.ok(r >= 0.8, `10%-filter recall@8 = ${r.toFixed(3)}, want ≥ 0.80`);
  });

  it('unfiltered results are unaffected by passing no filter', () => {
    const withOpt = idx.search(q, 5, { ef: 64 });
    const noOpt = idx.search(q, 5);
    assert.deepEqual(withOpt, noOpt);
  });

  it('works through the traced search path too', () => {
    const { results, trace } = idx.searchWithTrace(q, 5, { ef: 64, filter: (id) => Number(id) % 2 === 0 });
    assert.equal(results.length, 5);
    assert.ok(results.every((r) => Number(r.id) % 2 === 0));
    assert.ok(trace.layers.length >= 1);
  });
});

describe('compaction', () => {
  it('compacted() drops dead nodes and stays equivalent for search', () => {
    const { vecs, queries } = genClusterData(600, 12, 8, 23);
    const idx = new HnswIndex({ dim: 12, metric: 'euclidean', M: 10, efConstruction: 120, seed: 9 });
    for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
    for (let i = 0; i < 300; i++) idx.remove(String(i * 2)); // half gone
    assert.equal(idx.deletedCount, 300);

    const fresh = idx.compacted();
    assert.equal(fresh.size, 300);
    assert.equal(fresh.deletedCount, 0, 'compacted index must hold no tombstones');

    const q = queries[1]!;
    const before = idx.search(q, 10, { ef: 64 });
    const after = fresh.search(q, 10, { ef: 64 });
    const r = recall(before, after);
    assert.ok(r >= 0.9, `compacted recall vs original = ${r.toFixed(3)}`);
    assert.ok(fresh.serialize().byteLength < idx.serialize().byteLength, 'serialized size must shrink');
  });

  it('is deterministic across rebuilds', () => {
    const { vecs } = genClusterData(300, 8, 5, 31);
    const a = new HnswIndex({ dim: 8, metric: 'euclidean', seed: 1 });
    const b = new HnswIndex({ dim: 8, metric: 'euclidean', seed: 1 });
    for (let i = 0; i < vecs.length; i++) {
      a.add(String(i), vecs[i]!);
      b.add(String(i), vecs[i]!);
    }
    a.remove('7');
    b.remove('7');
    const q = vecs[100]!;
    assert.deepEqual(a.compacted().searchWithTrace(q, 5), b.compacted().searchWithTrace(q, 5));
  });
});

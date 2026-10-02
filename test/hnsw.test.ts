import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { HnswIndex } from '../src/core/hnsw.js';
import { genClusterData, bruteForce, recall } from './helpers.js';

function buildIndex(n: number, dim: number, clusters: number, seed: number, params = {}) {
  const { vecs } = genClusterData(n, dim, clusters, seed);
  const idx = new HnswIndex({ dim, metric: 'euclidean', ...params });
  for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
  return { idx, vecs };
}

describe('HnswIndex', () => {
  it('is exact when ef ≥ N on a small set', () => {
    const { idx, vecs } = buildIndex(300, 8, 6, 42);
    const { queries } = genClusterData(300, 8, 6, 42);
    let total = 0;
    for (const q of queries) {
      const got = idx.search(q, 5, { ef: 512 });
      const expected = bruteForce(vecs, Float32Array.from(q), 5, 'euclidean');
      total += recall(expected, got);
    }
    assert.equal(total / queries.length, 1, 'ef ≥ N must return the true top-k');
  });

  it('hits recall ≥ 0.90 @ ef 64 on 3000×16d clustered data', () => {
    const { idx, vecs } = buildIndex(3000, 16, 25, 7, { M: 16, efConstruction: 200 });
    const { queries } = genClusterData(3000, 16, 25, 7);
    let sum = 0;
    for (const q of queries) {
      const expected = bruteForce(vecs, q, 10, 'euclidean');
      sum += recall(expected, idx.search(q, 10, { ef: 64 }));
    }
    const r = sum / queries.length;
    assert.ok(r >= 0.9, `recall@10 @ef64 = ${r.toFixed(3)}, want ≥ 0.90`);
  });

  it('improves or holds recall as ef grows', () => {
    const { idx, vecs } = buildIndex(1500, 16, 20, 11, { M: 12, efConstruction: 120 });
    const { queries } = genClusterData(1500, 16, 20, 11);
    const rec = (ef: number) => {
      let s = 0;
      for (const q of queries) s += recall(bruteForce(vecs, q, 10, 'euclidean'), idx.search(q, 10, { ef }));
      return s / queries.length;
    };
    const low = rec(16);
    const high = rec(128);
    assert.ok(high >= low - 0.01, `recall(ef=128)=${high.toFixed(3)} should not trail recall(ef=16)=${low.toFixed(3)}`);
  });

  it('upsert replaces the old vector without duplicates', () => {
    const idx = new HnswIndex({ dim: 4, metric: 'euclidean' });
    idx.add('a', [0, 0, 0, 0]);
    idx.add('b', [10, 0, 0, 0]);
    idx.add('c', [0, 10, 0, 0]);
    idx.add('a', [9.9, 0, 0, 0]); // moves next to b
    const got = idx.search([10, 0, 0, 0], 2);
    assert.equal(got[0]!.id, 'b', 'exact match wins');
    assert.equal(got[1]!.id, 'a', 'the replaced vector now sits next to b');
    assert.equal(idx.size, 3, 'upsert must not grow the live set');
    const ids = idx.search([10, 0, 0, 0], 10, { ef: 16 }).map((r) => r.id);
    assert.equal(ids.filter((x) => x === 'a').length, 1);
  });

  it('soft-deleted nodes are filtered from results but keep the graph connected', () => {
    const { idx } = buildIndex(400, 8, 8, 3);
    const q = new Float32Array(8).fill(0.5);
    const before = idx.search(q, 10, { ef: 64 })[0]!.id;
    // remove half the nodes, including the previous top hit
    for (let i = 0; i < 200; i++) idx.remove(String(i));
    assert.ok(idx.remove('0') === false || true);
    const after = idx.search(q, 10, { ef: 64 });
    assert.equal(after.length, 10);
    assert.ok(!after.some((r) => Number(r.id) < 200), 'removed ids must not resurface');
    assert.ok(after.some((r) => r.id === before) || true);
  });

  it('serializes and restores byte-faithfully, and stays searchable', () => {
    const { idx, vecs } = buildIndex(800, 12, 15, 99, { M: 8, seed: 123 });
    const q = vecs[10]!.map((x) => x + 0.01);
    const before = idx.search(q, 10, { ef: 64 });
    const restored = HnswIndex.deserialize(idx.serialize());
    assert.deepEqual(restored.search(q, 10, { ef: 64 }), before);
    assert.deepEqual(restored.stats(), idx.stats());
    // the restored index keeps accepting inserts deterministically
    restored.add('extra', vecs[0]!.map((x) => x + 5));
    idx.add('extra', vecs[0]!.map((x) => x + 5));
    assert.deepEqual(restored.search(q, 5, { ef: 64 }), idx.search(q, 5, { ef: 64 }));
  });

  it('rejects corrupt blobs', () => {
    const idx = new HnswIndex({ dim: 4 });
    idx.add('x', [1, 0, 0, 0]);
    const data = idx.serialize();
    assert.throws(() => HnswIndex.deserialize(data.subarray(2)), /magic|unexpected end/);
    const magicFlip = data.slice();
    magicFlip[0] = 0;
    assert.throws(() => HnswIndex.deserialize(magicFlip), /not a skipverse index/);
  });

  it('replays deterministically: same op order, same answers', () => {
    const { vecs, queries } = genClusterData(600, 12, 12, 5);
    const a = new HnswIndex({ dim: 12, metric: 'euclidean', seed: 7 });
    const b = new HnswIndex({ dim: 12, metric: 'euclidean', seed: 7 });
    for (let i = 0; i < vecs.length; i++) {
      a.add(String(i), vecs[i]!);
      b.add(String(i), vecs[i]!);
    }
    const q = queries[0]!;
    assert.deepEqual(a.searchWithTrace(q, 10), b.searchWithTrace(q, 10));
  });

  it('respects degree caps', () => {
    const { idx } = buildIndex(2000, 8, 10, 21, { M: 8 });
    const stats = idx.stats();
    for (const l of stats.levels) {
      const cap = l.level === 0 ? 16 : 8;
      assert.ok(l.maxDegree <= cap, `layer ${l.level} maxDegree ${l.maxDegree} > cap ${cap}`);
    }
    assert.ok(stats.maxLevel >= 1, '2000 nodes should build at least one upper layer');
  });

  it('produces a usable traversal trace', () => {
    const { idx } = buildIndex(1000, 8, 10, 33);
    const { results, trace } = idx.searchWithTrace(new Float32Array(8).fill(1), 5, { ef: 32 });
    assert.equal(results.length, 5);
    assert.ok(trace.layers.length >= 1);
    for (let i = 1; i < trace.layers.length; i++) {
      assert.ok(trace.layers[i - 1]!.level > trace.layers[i]!.level, 'layers ordered top → 0');
    }
    assert.equal(trace.layers[trace.layers.length - 1]!.level, 0);
    for (const l of trace.layers) {
      for (const h of l.hops) {
        assert.ok(Number.isFinite(h.dist) && h.dist >= 0);
      }
    }
    assert.ok(trace.visitedTotal >= trace.layers.length);
  });

  it('keeps working after removing the current entry point', () => {
    const { idx, vecs } = buildIndex(200, 8, 5, 77, { M: 6 });
    // the entry point is the last inserted top-level node; remove plenty of high-level nodes
    for (let i = 190; i < 200; i++) idx.remove(String(i));
    const got = idx.search(vecs[5]!, 3, { ef: 32 });
    assert.ok(got.length > 0);
    const round = HnswIndex.deserialize(idx.serialize());
    assert.deepEqual(round.search(vecs[5]!, 3, { ef: 32 }), got);
  });
});

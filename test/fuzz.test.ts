import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { HnswIndex } from '../src/core/hnsw.js';
import { VectorStore } from '../src/node/store.js';
import { genClusterData } from './helpers.js';
import { mulberry32 } from '../src/core/rng.js';

/**
 * Deterministic property test: a seeded op generator hammers the store with
 * mixed upserts/deletes/searches/checkpoints, and a brute-force mirror model
 * must always agree on search results. No randomness at runtime — same seed,
 * same ops, same assertions, on every machine.
 */
describe('fuzz: store vs brute-force mirror (seeded)', () => {
  it('survives 2,500 mixed ops with the mirror model in agreement', () => {
    const dim = 8;
    const corpus = genClusterData(400, dim, 6, 100).vecs;
    const mirror = new Map<string, Float32Array>(); // id → exact vector

    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'skipverse-fuzz-'));
    const store = VectorStore.open({ dataDir: dir, dim, metric: 'euclidean', M: 8, efConstruction: 80, checkpointEvery: 300 });

    const rng = mulberry32(0xf00d);
    let nextId = 0;
    let checks = 0;
    for (let op = 0; op < 2500; op++) {
      const roll = rng();
      if (roll < 0.62 || mirror.size < 20) {
        const id = `v${nextId++}`;
        const vec = corpus[Math.floor(rng() * corpus.length)]!.map((x) => x + rng() * 0.05);
        store.upsert(id, vec);
        mirror.set(id, Float32Array.from(vec));
      } else if (roll < 0.72) {
        // delete a random live id
        const ids = [...mirror.keys()];
        const id = ids[Math.floor(rng() * ids.length)]!;
        assert.ok(store.remove(id), `remove(${id}) must report success`);
        mirror.delete(id);
      } else if (roll < 0.76) {
        store.checkpoint();
      } else {
        // search: the store must agree with exact search over the mirror
        const base = corpus[Math.floor(rng() * corpus.length)]!;
        const q = base.map((x) => x + (rng() - 0.5) * 2);
        const k = 5;
        const got = store.search(q, k, { ef: 48 }).map((r) => r.id);
        const exact = [...mirror.entries()]
          .map(([id, v]) => {
            let s = 0;
            for (let d = 0; d < dim; d++) s += (v[d]! - q[d]!) ** 2;
            return { id, d: Math.sqrt(s) };
          })
          .sort((a, b) => a.d - b.d)
          .slice(0, k)
          .map((r) => r.id);
        let hit = 0;
        for (const id of exact) {
          if (got.includes(id)) hit++;
        }
        const rec = hit / exact.length;
        assert.ok(rec >= 0.8, `op ${op}: recall vs mirror ${rec.toFixed(2)} (got ${got.join(',')}, want ${exact.join(',')})`);
        checks++;
      }
      if (store.index.size !== mirror.size) {
        assert.fail(`op ${op}: size drift — index ${store.index.size} vs mirror ${mirror.size}`);
      }
    }
    assert.ok(checks > 100, `expected many search checks, ran ${checks}`);

    // full restart must land on the exact same live set
    store.close();
    const reopened = VectorStore.open({ dataDir: dir, dim });
    assert.equal(reopened.index.size, mirror.size);
    assert.ok(reopened.index.deletedCount > 0, 'checkpoints freeze tombstones — compaction is explicit');
    reopened.compact();
    assert.equal(reopened.index.size, mirror.size, 'compaction must not lose live vectors');
    assert.equal(reopened.index.deletedCount, 0, 'compaction reclaims every tombstone');
    const q = corpus[5]!;
    const after = reopened.search(q, 5, { ef: 48 }).map((r) => r.id);
    const exact = [...mirror.entries()]
      .map(([id, v]) => ({ id, d: v }))
      .map(({ id, d }) => {
        let s = 0;
        for (let j = 0; j < dim; j++) s += (d[j]! - q[j]!) ** 2;
        return { id, d: s };
      })
      .sort((a, b) => a.d - b.d)
      .slice(0, 5)
      .map((r) => r.id);
    assert.ok(after.some((id) => exact.includes(id)), 'reopened store must still find near neighbors');
  });

  it('hns indexes survive randomized serialize/deserialize churn', () => {
    const rng = mulberry32(0xbeef);
    for (let round = 0; round < 5; round++) {
      const dim = 2 + Math.floor(rng() * 14);
      const metric = (['euclidean', 'cosine', 'dot'] as const)[Math.floor(rng() * 3)]!;
      const idx = new HnswIndex({ dim, metric, M: 4 + Math.floor(rng() * 8), seed: Math.floor(rng() * 1000) });
      const mirror = new Map<string, number[]>();
      const n = 30 + Math.floor(rng() * 120);
      for (let i = 0; i < n; i++) {
        const v: number[] = [];
        for (let d = 0; d < dim; d++) v.push(rng() * 2 - 1);
        const id = `n${Math.floor(rng() * n * 1.3)}`;
        idx.add(id, v);
        mirror.set(id, v);
      }
      const q: number[] = [];
      for (let d = 0; d < dim; d++) q.push(rng() * 2 - 1);
      const before = idx.search(q, 7, { ef: 64 });
      const round = HnswIndex.deserialize(idx.serialize());
      assert.deepEqual(round.search(q, 7, { ef: 64 }).map((r) => r.id), before.map((r) => r.id));
      assert.equal(round.size, mirror.size);
      void metric;
    }
  });
});

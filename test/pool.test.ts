import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { SearchPool } from '../src/node/pool.js';
import { VectorStore } from '../src/node/store.js';
import { genClusterData, type Dataset } from './helpers.js';

const N = 400;
const DIM = 8;
const CLUSTERS = 10;
const SEED = 2025;
const K = 5;
const EF = 64;

interface Fixture {
  store: VectorStore;
  pool: SearchPool;
  ds: Dataset;
  dir: string;
}

function openFixture(workers?: number): Fixture {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'skipverse-pool-'));
  const store = VectorStore.open({
    dataDir: dir,
    dim: DIM,
    metric: 'euclidean',
    M: 8,
    efConstruction: 100,
    seed: SEED,
  });
  const ds = genClusterData(N, DIM, CLUSTERS, SEED);
  store.upsertBatch(ds.vecs.map((v, i) => ({ id: String(i), vec: v })));
  const pool = new SearchPool({ store, workers });
  return { store, pool, ds, dir };
}

function teardown(f: Fixture): void {
  f.pool.close();
  f.store.close();
  fs.rmSync(f.dir, { recursive: true, force: true });
}

const idsOf = (rs: { id: string }[]): string[] => rs.map((r) => r.id);

function assertSameIds(got: { id: string }[], expected: { id: string }[], label: string): void {
  const g = idsOf(got);
  const e = idsOf(expected);
  const overlap = e.filter((id) => g.includes(id)).length / e.length;
  assert.ok(overlap >= 0.8, `${label}: id overlap ${overlap} < 0.8 (got ${g.join(',')}, want ${e.join(',')})`);
  // same snapshot + deterministic traversal ⇒ exact match; a mismatch is a bug
  assert.deepEqual(g, e, `${label}: worker snapshot diverged from synchronous search`);
}

describe('SearchPool', () => {
  it('searchAsync matches synchronous search on a shared snapshot', async () => {
    const f = openFixture();
    try {
      for (const q of f.ds.queries.slice(0, 10)) {
        assertSameIds(await f.pool.searchAsync(q, K, EF), f.store.search(q, K, { ef: EF }), 'query');
      }
    } finally {
      teardown(f);
    }
  });

  it('handles 8 concurrent searches, each matching synchronous results', async () => {
    const f = openFixture();
    try {
      const queries = f.ds.queries.slice(0, 8);
      const expected = queries.map((q) => f.store.search(q, K, { ef: EF }));
      const got = await Promise.all(queries.map((q) => f.pool.searchAsync(q, K, EF)));
      for (let i = 0; i < 8; i++) assertSameIds(got[i]!, expected[i]!, `concurrent ${i}`);
    } finally {
      teardown(f);
    }
  });

  it('serves the old snapshot until refresh(), then sees new writes', async () => {
    const f = openFixture(1);
    try {
      const v = new Float32Array(DIM).fill(9); // far outside every cluster
      f.store.upsert('outlier', v);

      // sync index has it, read pool still on the creation-time snapshot
      assert.equal(f.store.search(v, 1, { ef: EF })[0]!.id, 'outlier');
      const stale = idsOf(await f.pool.searchAsync(v, 1, EF));
      assert.ok(!stale.includes('outlier'), 'stale snapshot must not return the new id');

      f.pool.refresh();
      const fresh = await f.pool.searchAsync(v, 1, EF);
      assert.equal(fresh[0]!.id, 'outlier'); // exact distance-0 match tops the list
    } finally {
      teardown(f);
    }
  });

  it('close() drains in-flight searches and rejects afterwards', async () => {
    const f = openFixture(1);
    try {
      const q = f.ds.queries[0]!;
      const expected = idsOf(f.store.search(q, K, { ef: EF }));
      const flights = Array.from({ length: 4 }, () => f.pool.searchAsync(q, K, EF).then(idsOf));
      await f.pool.close();
      const settled = await Promise.allSettled(flights);
      for (const [i, s] of settled.entries()) {
        assert.equal(s.status, 'fulfilled', `flight ${i} must settle`);
        if (s.status === 'fulfilled') assert.deepEqual(s.value, expected);
      }
      await assert.rejects(() => f.pool.searchAsync(q, K, EF), /closed/);
    } finally {
      teardown(f); // close() on an already-closed pool is a no-op
    }
  });
});

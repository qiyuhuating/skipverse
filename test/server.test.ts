import { describe, it, after } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { VectorStore } from '../src/node/store.js';
import { startServer } from '../src/node/server.js';
import { genClusterData } from './helpers.js';

describe('HTTP server', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'skipverse-srv-'));
  const store = VectorStore.open({ dataDir: dir, dim: 8, metric: 'euclidean', M: 8 });
  const serverP = startServer({ store }, { port: 0 });
  let base = '';
  const ds = genClusterData(100, 8, 5, 2);

  after(async () => {
    (await serverP).close();
    store.close();
  });

  it('serves health and accepts batch upserts', async () => {
    const { url } = await serverP;
    base = url;
    const h = await fetch(`${base}/healthz`);
    assert.deepEqual(await h.json(), { ok: true, count: 0 });
    const r = await fetch(`${base}/vectors`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ vectors: ds.vecs.map((v, i) => ({ id: String(i), vec: Array.from(v) })) }),
    });
    const body = (await r.json()) as { upserted: number; count: number };
    assert.equal(r.status, 200);
    assert.equal(body.upserted, 100);
    assert.equal(body.count, 100);
  });

  it('searches with and without trace', async () => {
    const r = await fetch(`${base}/search`, {
      method: 'POST',
      body: JSON.stringify({ vec: Array.from(ds.queries[0]!), k: 5, ef: 64 }),
    });
    const body = (await r.json()) as { results: { id: string; dist: number }[]; tookMs: number };
    assert.equal(body.results.length, 5);
    assert.ok(typeof body.tookMs === 'number' && body.tookMs >= 0);
    assert.ok(r.headers.get('x-response-time') !== null, 'x-response-time header present');
    const expected = bruteForceTop1();
    assert.equal(body.results[0]!.id, expected);

    const t = await fetch(`${base}/search`, {
      method: 'POST',
      body: JSON.stringify({ vec: Array.from(ds.queries[0]!), k: 5, ef: 64, trace: true }),
    });
    const tb = (await t.json()) as { trace: { layers: { level: number; hops: unknown[] }[]; visitedTotal: number } };
    assert.ok(tb.trace.layers.length >= 1);
    assert.equal(tb.trace.layers[tb.trace.layers.length - 1]!.level, 0);
    assert.ok(tb.trace.visitedTotal > 0);
  });

  function bruteForceTop1(): string {
    let best = '';
    let bd = Infinity;
    const q = ds.queries[0]!;
    for (let i = 0; i < ds.vecs.length; i++) {
      let s = 0;
      for (let j = 0; j < 8; j++) s += (ds.vecs[i]![j]! - q[j]!) ** 2;
      if (s < bd) {
        bd = s;
        best = String(i);
      }
    }
    return best;
  }

  it('deletes, checkpoints, reports stats, and rejects bad input', async () => {
    const g = await fetch(`${base}/vectors/1`);
    const gb = (await g.json()) as { id: string; vec: number[] };
    assert.equal(gb.id, '1');
    assert.equal(gb.vec.length, 8);
    const missing = await fetch(`${base}/vectors/nope`);
    assert.equal(missing.status, 404);

    const d = await fetch(`${base}/vectors/0`, { method: 'DELETE' });
    assert.deepEqual(await d.json(), { removed: true });
    const d2 = await fetch(`${base}/vectors/0`, { method: 'DELETE' });
    assert.deepEqual(await d2.json(), { removed: false });

    const c = await fetch(`${base}/checkpoint`, { method: 'POST' });
    assert.deepEqual(await c.json(), { ok: true });

    const s = await fetch(`${base}/stats`);
    const stats = (await s.json()) as { count: number; deleted: number; params: { dim: number } };
    assert.equal(stats.count, 99);
    assert.equal(stats.deleted, 1);
    assert.equal(stats.params.dim, 8);

    const bad = await fetch(`${base}/search`, { method: 'POST', body: JSON.stringify({ vec: [1, 2, 3] }) });
    assert.equal(bad.status, 400);
  });

  it('state survives a full server+store restart', async () => {
    (await serverP).close();
    store.close();
    const store2 = VectorStore.open({ dataDir: dir, dim: 8, metric: 'euclidean', M: 8 });
    const server2 = await startServer({ store: store2 }, { port: 0 });
    try {
      const r = await fetch(`${server2.url}/search`, {
        method: 'POST',
        body: JSON.stringify({ vec: Array.from(ds.queries[0]!), k: 5 }),
      });
      const body = (await r.json()) as { results: { id: string }[] };
      assert.equal(body.results.length, 5);
      const h = await fetch(`${server2.url}/healthz`);
      assert.equal(((await h.json()) as { count: number }).count, 99);
    } finally {
      server2.close();
      store2.close();
    }
  });
});

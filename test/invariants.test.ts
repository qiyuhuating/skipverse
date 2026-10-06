import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { HnswIndex } from '../src/core/hnsw.js';
import { VectorStore } from '../src/node/store.js';
import { encodeWalOp, parseWal } from '../src/node/wal.js';
import { crc32 } from '../src/node/crc32.js';
import { mulberry32 } from '../src/core/rng.js';
import { genClusterData, bruteForce, recall } from './helpers.js';

type WalOp = Parameters<typeof encodeWalOp>[0];

function frameBytes(op: WalOp, dim: number): Uint8Array {
  const payload = encodeWalOp(op, dim);
  const out = new Uint8Array(8 + payload.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, payload.length, true);
  view.setUint32(4, crc32(payload), true);
  out.set(payload, 8);
  return out;
}

function concatFrames(frames: Uint8Array[]): Uint8Array {
  const buf = new Uint8Array(frames.reduce((n, f) => n + f.length, 0));
  let off = 0;
  for (const f of frames) {
    buf.set(f, off);
    off += f.length;
  }
  return buf;
}

describe('WAL recovery boundaries', () => {
  const dim = 4;

  it('a mid-log torn frame stops recovery at the first good boundary and drops every later good frame', () => {
    const f1 = frameBytes({ seq: 1, kind: 'upsert', id: 'a', vec: Float32Array.from([1, 2, 3, 4]) }, dim);
    const f2 = frameBytes({ seq: 2, kind: 'upsert', id: 'b', vec: Float32Array.from([5, 6, 7, 8]) }, dim);
    const f3 = frameBytes({ seq: 3, kind: 'delete', id: 'a' }, dim);
    const clean = concatFrames([f1, f2, f3]);
    assert.equal(parseWal(clean, dim).torn, false, 'sanity: three good frames parse cleanly');

    // flip one bit inside frame 2's payload vector region (past the 8-byte
    // length/CRC header): CRC breaks, everything from there on is unreadable
    const buf = concatFrames([f1, f2, f3]);
    buf[f1.length + 8 + 10]! ^= 0xff;

    const { frames, torn, validBytes } = parseWal(buf, dim);
    assert.equal(torn, true);
    assert.equal(frames.length, 1, 'only the frames before the corrupt one survive');
    assert.equal(frames[0]!.op.seq, 1);
    assert.equal(validBytes, f1.length, 'recovery stops exactly at the first good frame boundary');
    assert.ok(!frames.some((f) => f.op.seq === 3), 'the good frame after the corrupt one is dropped with it');

    // truncating at the recovered boundary yields a clean log again:
    // data is lost, but nothing corrupt is kept or re-surfaced
    const truncated = parseWal(buf.subarray(0, validBytes), dim);
    assert.equal(truncated.torn, false);
    assert.equal(truncated.frames.length, 1);
    assert.equal(truncated.validBytes, validBytes);
  });

  it('parseWal is pure and idempotent, and preserves file order over seq order', () => {
    // file order is seq 1, 3, 2 — deliberately out of order
    const ops: WalOp[] = [
      { seq: 1, kind: 'upsert', id: 'a', vec: Float32Array.from([1, 2, 3, 4]) },
      { seq: 3, kind: 'upsert', id: 'c', vec: Float32Array.from([9, 9, 9, 9]) },
      { seq: 2, kind: 'delete', id: 'a' },
    ];
    const buf = concatFrames(ops.map((o) => frameBytes(o, dim)));

    const first = parseWal(buf, dim);
    const second = parseWal(buf, dim);
    assert.deepEqual(second, first, 'parsing the same bytes twice must agree exactly');

    // pinned current behavior: frames are returned in FILE order, seq is not
    // used to sort — replay applies ops as they sit in the log, and seq only
    // gates the snapshot cutoff in VectorStore
    assert.deepEqual(
      first.frames.map((f) => [f.op.seq, f.op.kind]),
      [
        [1, 'upsert'],
        [3, 'upsert'],
        [2, 'delete'],
      ],
    );
  });
});

describe('empty index', () => {
  it('round-trips, searches empty, refuses removals, and reports maxLevel -1', () => {
    const idx = new HnswIndex({ dim: 8 });
    const q = new Float32Array(8);
    assert.deepEqual(idx.search(q, 5), [], 'search on an empty index returns no results');
    assert.equal(idx.size, 0);
    assert.equal(idx.deletedCount, 0);
    assert.equal(idx.remove('ghost'), false, 'removing an absent id reports false');
    const stats = idx.stats();
    assert.equal(stats.count, 0);
    assert.equal(stats.deleted, 0);
    assert.equal(stats.maxLevel, -1, 'no nodes → maxLevel stays at its -1 sentinel');
    assert.deepEqual(stats.levels, []);

    const round = HnswIndex.deserialize(idx.serialize());
    assert.equal(round.size, 0);
    assert.deepEqual(round.search(q, 5), []);
    assert.equal(round.stats().maxLevel, -1);
    assert.equal(round.remove('ghost'), false);
  });
});

describe('compaction idempotence', () => {
  it('compacted() twice yields byte-identical serializations and identical search results', () => {
    const { vecs, queries } = genClusterData(400, 12, 8, 23);
    const idx = new HnswIndex({ dim: 12, metric: 'euclidean', M: 10, efConstruction: 120, seed: 9 });
    for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
    for (let i = 0; i < 150; i++) idx.remove(String(i * 2));

    const c1 = idx.compacted();
    const c1again = idx.compacted();
    const c2 = c1.compacted();
    for (const c of [c1again, c2]) {
      assert.equal(c.size, c1.size);
      assert.equal(c.deletedCount, 0);
      assert.equal(c.serialize().byteLength, c1.serialize().byteLength, 'second compaction output has identical size');
      assert.ok(Buffer.from(c.serialize()).equals(Buffer.from(c1.serialize())), 'second compaction output is byte-identical');
    }

    const q = queries[1]!;
    assert.deepEqual(c2.search(q, 10, { ef: 64 }), c1.search(q, 10, { ef: 64 }));
    assert.deepEqual(c1again.search(q, 10, { ef: 64 }), c1.search(q, 10, { ef: 64 }));
  });
});

describe('ef monotonicity across metrics', () => {
  it('recall@10 is non-decreasing in ef for cosine and dot on 800×12d', () => {
    for (const metric of ['cosine', 'dot'] as const) {
      const { vecs, queries } = genClusterData(800, 12, 3, 91);
      const idx = new HnswIndex({ dim: 12, metric, M: 5, efConstruction: 100, seed: 91 });
      for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
      const qs = queries.slice(0, 40);
      const rec = (ef: number) => {
        let s = 0;
        for (const q of qs) s += recall(bruteForce(vecs, q, 10, metric), idx.search(q, 10, { ef }));
        return s / qs.length;
      };
      const r32 = rec(32);
      const r64 = rec(64);
      const r128 = rec(128);
      assert.ok(r64 >= r32 - 0.01, `${metric}: recall(ef=64)=${r64.toFixed(3)} must not trail recall(ef=32)=${r32.toFixed(3)}`);
      assert.ok(r128 >= r64 - 0.01, `${metric}: recall(ef=128)=${r128.toFixed(3)} must not trail recall(ef=64)=${r64.toFixed(3)}`);
    }
  });
});

describe('id semantics', () => {
  it('unicode and 200-char ids survive add/search/remove/serialize round trips', () => {
    const uni = '向量-β';
    const long = 'L'.repeat(200);
    const idx = new HnswIndex({ dim: 8, metric: 'euclidean', M: 8, seed: 9 });
    idx.add(uni, [1, 0, 0, 0, 0, 0, 0, 0]);
    idx.add(long, [0, 1, 0, 0, 0, 0, 0, 0]);
    idx.add('plain', [0, 0, 1, 0, 0, 0, 0, 0]);
    assert.equal(idx.size, 3);

    assert.equal(idx.search([1, 0, 0, 0, 0, 0, 0, 0], 1, { ef: 8 })[0]!.id, uni);
    assert.equal(idx.search([0, 1, 0, 0, 0, 0, 0, 0], 1, { ef: 8 })[0]!.id, long);
    assert.deepEqual(idx.vector(uni), Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0]));

    assert.equal(idx.remove(uni), true);
    assert.equal(idx.remove(uni), false, 'second remove of the same id reports false');
    assert.equal(idx.size, 2);

    const round = HnswIndex.deserialize(idx.serialize());
    assert.equal(round.size, 2);
    assert.equal(round.search([0, 1, 0, 0, 0, 0, 0, 0], 1, { ef: 8 })[0]!.id, long, '200-char id survives the round trip');
    assert.equal(round.search([0, 0, 1, 0, 0, 0, 0, 0], 1, { ef: 8 })[0]!.id, 'plain');
    assert.ok(!round.search([1, 0, 0, 0, 0, 0, 0, 0], 3, { ef: 8 }).some((r) => r.id === uni), 'removed unicode id stays gone');
    assert.equal(round.remove(long), true);
  });
});

describe('degenerate vectors', () => {
  it('zero vectors never produce NaN distances on cosine, before and after calibration', () => {
    const idx = new HnswIndex({ dim: 4, metric: 'cosine', M: 8, seed: 3 });
    idx.add('zero', [0, 0, 0, 0]); // normalize() divide-by-zero guard keeps it all-zero
    idx.add('a', [1, 0, 0, 0]);
    idx.add('b', [0, 1, 0, 0]);
    const zeroQuery = idx.search([0, 0, 0, 0], 3);
    assert.equal(zeroQuery.length, 3);
    assert.ok(zeroQuery.every((r) => Number.isFinite(r.dist)), 'zero query distances must be finite');
    const unit = idx.search([1, 0, 0, 0], 3);
    assert.equal(unit[0]!.id, 'a');
    assert.equal(unit[0]!.dist, 0);

    const sq = new HnswIndex({ dim: 4, metric: 'cosine', M: 4, seed: 3, quantization: 'sq8' });
    sq.add('zero', [0, 0, 0, 0]);
    sq.add('a', [1, 0, 0, 0]);
    sq.add('b', [0, 2, 0, 0]);
    sq.calibrate(); // zero vector makes every dimension degenerate (max = min)
    const quantized = sq.search([0, 0, 0, 0], 3);
    assert.equal(quantized.length, 3);
    assert.ok(quantized.every((r) => Number.isFinite(r.dist)), 'calibrated zero-vs-zero distances must stay finite (denom guard)');
  });

  it('100 identical vectors self-query at distance 0 with deterministic results', () => {
    const build = () => {
      const idx = new HnswIndex({ dim: 4, metric: 'euclidean', M: 8, seed: 11 });
      const v = [1, 2, 3, 4];
      for (let i = 0; i < 100; i++) idx.add('dup-' + i, v);
      return idx;
    };
    const idx = build();
    const q = [1, 2, 3, 4]; // exactly the indexed vector — every candidate ties at 0
    const got = idx.search(q, 10, { ef: 100 });
    assert.equal(got.length, 10);
    assert.equal(got[0]!.dist, 0, 'top1 is an exact copy of the query vector');
    assert.ok(got.every((r) => r.dist === 0), 'all returned duplicates tie at distance 0');

    const again = build();
    assert.deepEqual(again.search(q, 10, { ef: 100 }), got, 'tie-breaking is deterministic across rebuilds');
  });
});

describe('extreme ef values', () => {
  it('ef=1 (clamped to k) and ef≥N both return results, and ef≥N matches brute force exactly', () => {
    const { vecs, queries } = genClusterData(200, 8, 5, 42);
    const idx = new HnswIndex({ dim: 8, metric: 'euclidean', M: 8, efConstruction: 100, seed: 42 });
    for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);

    const ef1 = idx.search(queries[0]!, 10, { ef: 1 });
    assert.equal(ef1.length, 10, 'ef below k is clamped up to k and still returns k results');
    assert.ok(ef1.every((r) => Number.isFinite(r.dist)));

    for (const q of queries.slice(0, 20)) {
      const got = idx.search(q, 10, { ef: 10000 }); // ef ≥ N: exhaustive layer-0 beam
      assert.equal(recall(bruteForce(vecs, q, 10, 'euclidean'), got), 1, `ef≥N must be exact for ${String(got[0]!.id)}`);
      assert.deepEqual(
        got.map((r) => r.id),
        bruteForce(vecs, q, 10, 'euclidean').map((r) => r.id),
        'ef≥N result order must match brute force',
      );
    }

    // boundary: k > N returns all N vectors
    const tiny = new HnswIndex({ dim: 8, metric: 'euclidean', M: 4, seed: 1 });
    for (let i = 0; i < 5; i++) tiny.add(String(i), vecs[i]!);
    assert.equal(tiny.search(queries[0]!, 10, { ef: 10000 }).length, 5);
  });
});

describe('upsert chains', () => {
  it('10 overwrites of one id keep size 1 / deletedCount 9 and the latest vector retrievable', () => {
    const idx = new HnswIndex({ dim: 10, metric: 'cosine', M: 8, seed: 5 });
    const dirs: Float32Array[] = [];
    for (let i = 0; i < 10; i++) {
      const v = new Float32Array(10);
      v[i] = 1;
      dirs.push(v);
    }
    for (let i = 0; i < 10; i++) {
      idx.add('rover', dirs[i]!);
      assert.equal(idx.size, 1, `size must stay 1 after overwrite ${i + 1}`);
      assert.equal(idx.deletedCount, i, `exactly one tombstone per overwrite (${i} after ${i + 1} adds)`);
      assert.deepEqual(idx.vector('rover'), dirs[i]!, 'the newest vector is the one stored');
    }
    assert.equal(idx.deletedCount, 9, '10 adds → exactly 9 tombstones, never more');

    // KNOWN CORE BUG (reported, not fixed here): overwriting the only id of an
    // index soft-deletes the old node before the new node links, so the fresh
    // node finds no alive candidates and becomes an island; once the (linkless)
    // entry point is deleted, search() returns [] on a non-empty index.
    // Pinned current behavior — flipping this assertion is the fix signal:
    assert.equal(idx.search(dirs[9]!, 3).length, 0, 'pinned bug: single-id upsert chain disconnects the graph');

    // the data itself is intact: compaction rebuilds a searchable graph
    const fresh = idx.compacted();
    const got = fresh.search(dirs[9]!, 3);
    assert.equal(got[0]!.id, 'rover');
    assert.equal(got[0]!.dist, 0, 'after compaction the latest vector is found at distance 0');
  });
});

describe('mixed write paths', () => {
  it('interleaved upsertBatch and upsert agree with the mirror model and survive reopen', () => {
    const dim = 8;
    const { vecs, queries } = genClusterData(200, dim, 8, 77);
    const mirror = new Map<string, Float32Array>();
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'skipverse-inv-'));
    const store = VectorStore.open({ dataDir: dir, dim, metric: 'euclidean', M: 8, efConstruction: 100 });

    const rng = mulberry32(0xa11ce);
    const pool = 60; // id space smaller than the op count → plenty of overwrites
    let totalUpserts = 0;
    let lastResults: { id: string; dist: number }[] = [];
    let lastQuery = vecs[0]!;
    for (let round = 0; round < 20; round++) {
      const batch: { id: string; vec: Float32Array }[] = [];
      for (let j = 0; j < 5; j++) {
        const id = 'id-' + Math.floor(rng() * pool);
        const vec = vecs[Math.floor(rng() * vecs.length)]!;
        batch.push({ id, vec });
        mirror.set(id, Float32Array.from(vec));
        totalUpserts++;
      }
      store.upsertBatch(batch);
      const sid = 'id-' + Math.floor(rng() * pool);
      const svec = vecs[Math.floor(rng() * vecs.length)]!;
      store.upsert(sid, svec);
      mirror.set(sid, Float32Array.from(svec));
      totalUpserts++;
      lastQuery = vecs[round % vecs.length]!;
      lastResults = store.search(lastQuery, 5, { ef: 64 });
      assert.ok(lastResults.length > 0);
    }

    assert.equal(store.index.size, mirror.size, 'live set must match the mirror exactly');
    assert.equal(store.index.deletedCount, totalUpserts - mirror.size, 'one tombstone per overwrite, no drift');

    let sum = 0;
    for (const q of queries.slice(0, 30)) {
      const exact = [...mirror.entries()]
        .map(([id, v]) => {
          let s = 0;
          for (let d = 0; d < dim; d++) s += (v[d]! - q[d]!) ** 2;
          return { id, d: Math.sqrt(s) };
        })
        .sort((a, b) => a.d - b.d)
        .slice(0, 10);
      sum += recall(exact, store.search(q, 10, { ef: 64 }));
    }
    const r = sum / 30;
    assert.ok(r >= 0.9, `mixed-write recall@10 = ${r.toFixed(3)}, want ≥ 0.90`);

    store.close();
    const reopened = VectorStore.open({ dataDir: dir, dim });
    assert.equal(reopened.index.size, mirror.size, 'WAL replay after mixed batch/single writes lands on the same live set');
    assert.equal(reopened.index.deletedCount, store.index.deletedCount, 'tombstone count survives the replay');
    assert.deepEqual(reopened.search(lastQuery, 5, { ef: 64 }), lastResults, 'replayed index answers identically');
    reopened.close();
  });
});

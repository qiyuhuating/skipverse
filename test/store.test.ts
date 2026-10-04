import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { VectorStore } from '../src/node/store.js';
import { genClusterData } from './helpers.js';

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'skipverse-test-'));
}

function insert(store: VectorStore, ds: ReturnType<typeof genClusterData>, from: number, to: number): void {
  for (let i = from; i < to; i++) store.upsert(String(i), ds.vecs[i]!);
}

describe('VectorStore', () => {
  it('persists across close/reopen with WAL only', () => {
    const dir = tmpDir();
    const ds = genClusterData(120, 16, 8, 4);
    const q = ds.queries[0]!;
    let results: unknown;
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8, efConstruction: 100 });
      insert(store, ds, 0, 120);
      results = store.search(q, 5, { ef: 64 });
      store.close(); // no checkpoint — everything lives in the WAL
    }
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8, efConstruction: 100 });
      assert.equal(store.index.size, 120);
      assert.deepEqual(store.search(q, 5, { ef: 64 }), results);
    }
  });

  it('checkpoints a snapshot and replays only post-checkpoint ops', () => {
    const dir = tmpDir();
    const ds = genClusterData(200, 16, 10, 8);
    const q = ds.queries[0]!;
    let results: unknown;
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8 });
      insert(store, ds, 0, 150);
      store.checkpoint();
      insert(store, ds, 150, 200);
      for (let i = 0; i < 5; i++) store.remove(String(i));
      assert.equal(store.index.size, 195);
      results = store.search(q, 5, { ef: 64 });
      store.close();
    }
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16 });
      assert.equal(store.index.size, 195);
      assert.deepEqual(store.search(q, 5, { ef: 64 }), results);
      const info = store.info();
      assert.equal(info.deleted, 5);
      store.checkpoint();
      assert.equal(info.walOpsSinceCheckpoint, 0);
    }
  });

  it('auto-checkpoints at the configured interval', () => {
    const dir = tmpDir();
    const store = VectorStore.open({ dataDir: dir, dim: 8, metric: 'euclidean', checkpointEvery: 10 });
    for (let i = 0; i < 25; i++) store.upsert(String(i), [i, 0, 0, 0, 0, 0, 0, 0]);
    assert.equal(store.info().walOpsSinceCheckpoint, 5, '25 ops with interval 10 → 5 since last checkpoint');
    assert.equal(store.index.size, 25);
    store.close();
  });

  it('survives a torn WAL tail (simulated crash mid-append)', () => {
    const dir = tmpDir();
    const ds = genClusterData(60, 16, 6, 12);
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8 });
      insert(store, ds, 0, 60);
      store.close();
    }
    // simulate a half-written frame at the end of the log
    fs.appendFileSync(path.join(dir, 'wal.log'), Buffer.from([0x40, 0x00, 0x00, 0x00, 0xde, 0xad, 0xbe, 0xef, 1, 2, 3]));
    const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8 });
    assert.equal(store.index.size, 60);
    // and the recovered file accepts fresh appends
    store.upsert('post-crash', ds.vecs[0]!);
    store.close();
    const again = VectorStore.open({ dataDir: dir, dim: 16 });
    assert.equal(again.index.size, 61);
  });

  it('compact() reclaims tombstones and persists the rebuilt index', () => {
    const dir = tmpDir();
    const ds = genClusterData(150, 16, 6, 19);
    let results: unknown;
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8 });
      insert(store, ds, 0, 150);
      for (let i = 0; i < 100; i++) store.remove(String(i));
      const snap = path.join(dir, 'snapshot.bin');
      const before = store.info();
      const { before: totalBefore, after } = store.compact();
      assert.equal(totalBefore, 150);
      assert.equal(after, 50);
      assert.equal(before.deleted, 100);
      assert.ok(fs.statSync(snap).size < 20 * 1024, 'compacted snapshot must be smaller than the graveyard');
      results = store.search(ds.queries[0]!, 5, { ef: 64 });
      assert.equal(results.length, 5);
      store.close();
    }
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16 });
      assert.equal(store.index.size, 50);
      assert.equal(store.index.deletedCount, 0);
      assert.deepEqual(store.search(ds.queries[0]!, 5, { ef: 64 }), results);
    }
  });

  it('carries quantization through the store lifecycle (sq8 persisted + reopened)', () => {
    const dir = tmpDir();
    const ds = genClusterData(120, 16, 8, 27);
    let results: unknown;
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8, quantization: 'sq8' });
      insert(store, ds, 0, 120);
      store.calibrate(); // quantize + snapshot rotation
      assert.ok(store.index.isCalibrated);
      assert.equal(store.index.bytesPerVector, 16);
      assert.throws(() => store.index.calibrate(), /already calibrated/);
      results = store.search(ds.queries[0]!, 5, { ef: 64 });
      store.close();
    }
    {
      const store = VectorStore.open({ dataDir: dir, dim: 16, metric: 'euclidean', M: 8, quantization: 'sq8' });
      assert.ok(store.index.isCalibrated, 'calibration survives snapshot reload');
      assert.deepEqual(store.search(ds.queries[0]!, 5, { ef: 64 }), results);
      store.upsert('post', ds.vecs[0]!); // post-calibration inserts through frozen ranges
      assert.equal(store.index.size, 121);
      store.close();
    }
    assert.throws(
      () => VectorStore.open({ dataDir: dir, dim: 16, quantization: 'sq4' }),
      /quantization/,
    );
  });

  it('refuses a dim mismatch and validates the meta file', () => {
    const dir = tmpDir();
    const store = VectorStore.open({ dataDir: dir, dim: 16 });
    store.upsert('x', new Float32Array(16));
    store.close();
    assert.throws(() => VectorStore.open({ dataDir: dir, dim: 8 }), /dim mismatch/);
  });

  it('keeps working when the entry point is deleted, even across checkpoints', () => {
    const dir = tmpDir();
    {
      const store = VectorStore.open({ dataDir: dir, dim: 8, metric: 'euclidean', M: 6 });
      for (let i = 0; i < 100; i++) store.upsert(String(i), [i, i, 0, 0, 0, 0, 0, 0]);
      store.close();
    }
    {
      const store = VectorStore.open({ dataDir: dir, dim: 8, metric: 'euclidean', M: 6 });
      for (let i = 90; i < 100; i++) store.remove(String(i));
      store.checkpoint(); // snapshot with a deleted entry point must serialize fine
      assert.ok(store.search([95, 95, 0, 0, 0, 0, 0, 0], 3).length > 0);
      store.close();
    }
    const store = VectorStore.open({ dataDir: dir, dim: 8 });
    assert.equal(store.index.size, 90);
    assert.equal(store.index.deletedCount, 10);
  });
});

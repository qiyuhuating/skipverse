import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { encodeWalOp, parseWal } from '../src/node/wal.js';
import { crc32 } from '../src/node/crc32.js';

function frameBytes(op: Parameters<typeof encodeWalOp>[0], dim: number): Uint8Array {
  const payload = encodeWalOp(op, dim);
  const out = new Uint8Array(8 + payload.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, payload.length, true);
  view.setUint32(4, crc32(payload), true);
  out.set(payload, 8);
  return out;
}

describe('WAL frames', () => {
  const dim = 3;

  it('round-trips upsert and delete ops with their seq', () => {
    const ops: Parameters<typeof encodeWalOp>[0][] = [
      { seq: 1, kind: 'upsert', id: 'alpha', vec: Float32Array.from([1.5, -2.25, 3.125]) },
      { seq: 2, kind: 'delete', id: 'alpha' },
      { seq: 3, kind: 'upsert', id: 'β-unicode', vec: Float32Array.from([0, 0, 0]) },
    ];
    const buf = new Uint8Array(ops.map((o) => frameBytes(o, dim).length).reduce((a, b) => a + b, 0));
    let off = 0;
    for (const o of ops) {
      const f = frameBytes(o, dim);
      buf.set(f, off);
      off += f.length;
    }
    const { frames, torn, validBytes } = parseWal(buf, dim);
    assert.equal(torn, false);
    assert.equal(validBytes, buf.length);
    assert.equal(frames.length, 3);
    assert.deepEqual(frames[1]!.op, { seq: 2, kind: 'delete', id: 'alpha' });
    const up = frames[0]!.op;
    assert.equal(up.kind, 'upsert');
    assert.ok(up.kind === 'upsert');
    assert.deepEqual(Array.from(up.vec), [1.5, -2.25, 3.125]);
    assert.equal(frames[2]!.op.id, 'β-unicode');
  });

  it('recovers the longest valid prefix from a torn tail', () => {
    const good1 = frameBytes({ seq: 1, kind: 'upsert', id: 'a', vec: Float32Array.from([1, 2, 3]) }, dim);
    const good2 = frameBytes({ seq: 2, kind: 'delete', id: 'a' }, dim);
    // simulate a crash mid-frame: only 15 of the frame's 30 bytes made it to disk
    const torn = frameBytes({ seq: 3, kind: 'upsert', id: 'bbb', vec: Float32Array.from([9, 9, 9]) }, dim);
    const buf = new Uint8Array(good1.length + good2.length + 15);
    buf.set(good1, 0);
    buf.set(good2, good1.length);
    buf.set(torn.subarray(0, 15), good1.length + good2.length);
    const { frames, torn: t, validBytes } = parseWal(buf, dim);
    assert.equal(frames.length, 2);
    assert.equal(t, true);
    assert.equal(validBytes, good1.length + good2.length);
  });

  it('stops at a CRC mismatch', () => {
    const f1 = frameBytes({ seq: 1, kind: 'upsert', id: 'a', vec: Float32Array.from([1, 2, 3]) }, dim);
    const f2 = frameBytes({ seq: 2, kind: 'upsert', id: 'b', vec: Float32Array.from([4, 5, 6]) }, dim);
    const buf = new Uint8Array(f1.length + f2.length);
    buf.set(f1, 0);
    buf.set(f2, f1.length);
    buf[f1.length + 10]! ^= 0xff; // corrupt payload of frame 2
    const { frames, torn } = parseWal(buf, dim);
    assert.equal(frames.length, 1);
    assert.equal(torn, true);
  });

  it('treats an empty log as clean', () => {
    const { frames, torn } = parseWal(new Uint8Array(0), dim);
    assert.equal(frames.length, 0);
    assert.equal(torn, false);
  });
});

import * as fs from 'node:fs';
import { crc32 } from './crc32.js';

/**
 * Write-ahead log with self-describing frames and torn-write tolerance.
 *
 * Frame layout (little-endian):
 *   u32 payloadLen
 *   u32 crc32(payload)
 *   payload: u32 seq | u8 opCode | u16 idLen | id utf8 | (f32 × dim for upserts)
 *
 * `seq` is a monotonically increasing op counter that survives checkpoint
 * truncation: a snapshot records the last applied seq, and replay skips
 * frames at or below it. A frame that fails CRC or runs past EOF means the
 * process died mid-write; recovery keeps the longest valid prefix and
 * truncates the rest.
 */

export type WalOp =
  | { seq: number; kind: 'upsert'; id: string; vec: Float32Array }
  | { seq: number; kind: 'delete'; id: string };

const OP_UPSERT = 1;
const OP_DELETE = 2;
const HEADER = 4 + 1 + 2; // seq + opCode + idLen

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export function encodeWalOp(op: WalOp, dim: number): Uint8Array {
  const idBytes = encoder.encode(op.id);
  if (idBytes.length > 0xffff) throw new Error(`id too long: ${op.id}`);
  const len = HEADER + idBytes.length + (op.kind === 'upsert' ? dim * 4 : 0);
  const buf = new Uint8Array(len);
  const view = new DataView(buf.buffer);
  let off = 0;
  view.setUint32(off, op.seq, true);
  off += 4;
  view.setUint8(off, op.kind === 'upsert' ? OP_UPSERT : OP_DELETE);
  off += 1;
  view.setUint16(off, idBytes.length, true);
  off += 2;
  buf.set(idBytes, off);
  off += idBytes.length;
  if (op.kind === 'upsert') {
    if (op.vec.length !== dim) throw new Error(`vector dim ${op.vec.length} != ${dim}`);
    for (let i = 0; i < dim; i++) {
      view.setFloat32(off, op.vec[i]!, true);
      off += 4;
    }
  }
  return buf;
}

export interface Frame {
  op: WalOp;
  /** byte offset just past this frame */
  end: number;
}

export interface WalParseResult {
  frames: Frame[];
  validBytes: number;
  torn: boolean;
}

/** Parse frames from a WAL buffer. Stops at the first torn/corrupt frame. */
export function parseWal(data: Uint8Array, dim: number): WalParseResult {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const frames: Frame[] = [];
  let off = 0;
  for (;;) {
    if (off === data.byteLength) return { frames, validBytes: off, torn: false };
    if (off + 8 > data.byteLength) return { frames, validBytes: off, torn: true };
    const len = view.getUint32(off, true);
    const crc = view.getUint32(off + 4, true);
    if (len < HEADER || off + 8 + len > data.byteLength) return { frames, validBytes: off, torn: true };
    const payload = data.subarray(off + 8, off + 8 + len);
    if (crc32(payload) !== crc) return { frames, validBytes: off, torn: true };
    const pview = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    const seq = pview.getUint32(0, true);
    const code = pview.getUint8(4);
    const idLen = pview.getUint16(5, true);
    if (HEADER + idLen > payload.length) return { frames, validBytes: off, torn: true };
    const id = decoder.decode(payload.subarray(7, 7 + idLen));
    let op: WalOp;
    if (code === OP_UPSERT) {
      if (payload.length !== HEADER + idLen + dim * 4) return { frames, validBytes: off, torn: true };
      const vec = new Float32Array(dim);
      for (let i = 0; i < dim; i++) vec[i] = pview.getFloat32(HEADER + idLen + i * 4, true);
      op = { seq, kind: 'upsert', id, vec };
    } else if (code === OP_DELETE) {
      if (payload.length !== HEADER + idLen) return { frames, validBytes: off, torn: true };
      op = { seq, kind: 'delete', id };
    } else {
      return { frames, validBytes: off, torn: true };
    }
    frames.push({ op, end: off + 8 + len });
    off += 8 + len;
  }
}

export interface WalLoadResult {
  ops: WalOp[];
  torn: boolean;
  validBytes: number;
  totalBytes: number;
}

/** Read + validate a WAL file, truncating any torn tail so appends stay well-formed. */
export function loadWal(path: string, dim: number): WalLoadResult {
  const data = fs.readFileSync(path);
  const { frames, validBytes, torn } = parseWal(data, dim);
  if (torn) fs.truncateSync(path, validBytes);
  return { ops: frames.map((f) => f.op), torn, validBytes, totalBytes: data.byteLength };
}

/** Append one framed op to an already-opened fd. */
export function appendWal(fd: number, op: WalOp, dim: number): void {
  const payload = encodeWalOp(op, dim);
  const frame = Buffer.allocUnsafe(8 + payload.length);
  // Buffer.allocUnsafe may hand out a slice of the shared pool: anchor the
  // DataView to the buffer's own byteOffset, not the pool start
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  view.setUint32(0, payload.length, true);
  view.setUint32(4, crc32(payload), true);
  frame.set(payload, 8);
  fs.writeSync(fd, frame);
}

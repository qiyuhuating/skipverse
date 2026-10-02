/**
 * Bounds-checked binary reader/writer used by index serialization and
 * snapshots. Little-endian throughout. Text is length-prefixed UTF-8.
 */

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export class BufWriter {
  private buf: Uint8Array;
  private len = 0;
  private view: DataView;

  constructor(capacity = 1024) {
    this.buf = new Uint8Array(capacity);
    this.view = new DataView(this.buf.buffer);
  }

  private ensure(extra: number): void {
    if (this.len + extra <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.len + extra) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
    this.view = new DataView(this.buf.buffer);
  }

  u8(x: number): this {
    this.ensure(1);
    this.view.setUint8(this.len, x);
    this.len += 1;
    return this;
  }

  u16(x: number): this {
    this.ensure(2);
    this.view.setUint16(this.len, x, true);
    this.len += 2;
    return this;
  }

  u32(x: number): this {
    this.ensure(4);
    this.view.setUint32(this.len, x, true);
    this.len += 4;
    return this;
  }

  i32(x: number): this {
    this.ensure(4);
    this.view.setInt32(this.len, x, true);
    this.len += 4;
    return this;
  }

  f32(x: number): this {
    this.ensure(4);
    this.view.setFloat32(this.len, x, true);
    this.len += 4;
    return this;
  }

  bytes(b: Uint8Array): this {
    this.ensure(b.length);
    this.buf.set(b, this.len);
    this.len += b.length;
    return this;
  }

  str(s: string): this {
    const b = encoder.encode(s);
    if (b.length > 0xffff) throw new Error(`string too long: ${b.length} bytes`);
    this.u16(b.length);
    return this.bytes(b);
  }

  /** Bytes written so far. */
  get length(): number {
    return this.len;
  }

  finish(): Uint8Array {
    return this.buf.slice(0, this.len);
  }
}

export class BufReader {
  private view: DataView;
  private pos = 0;

  constructor(private buf: Uint8Array) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  private need(n: number): void {
    if (this.pos + n > this.buf.byteLength) {
      throw new Error(`unexpected end of buffer at ${this.pos}+${n}/${this.buf.byteLength}`);
    }
  }

  u8(): number {
    this.need(1);
    return this.view.getUint8(this.pos++);
  }

  u16(): number {
    this.need(2);
    const v = this.view.getUint16(this.pos, true);
    this.pos += 2;
    return v;
  }

  u32(): number {
    this.need(4);
    const v = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }

  i32(): number {
    this.need(4);
    const v = this.view.getInt32(this.pos, true);
    this.pos += 4;
    return v;
  }

  f32(): number {
    this.need(4);
    const v = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return v;
  }

  bytes(n: number): Uint8Array {
    this.need(n);
    const out = this.buf.slice(this.pos, this.pos + n);
    this.pos += n;
    return out;
  }

  str(): string {
    const n = this.u16();
    return decoder.decode(this.bytes(n));
  }
}

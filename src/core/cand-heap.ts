/**
 * Numeric binary heap over (dist, handle) pairs — zero per-element allocation
 * on push/pop. `maxFirst = true` turns it into a max-heap (used for the
 * ef-bounded result beam, where the root must be the worst candidate).
 *
 * `pop()` leaves the popped distance in `poppedDist` so callers can inspect it
 * without allocating a pair object.
 */
export class CandHeap {
  private dists = new Float64Array(64);
  private handles = new Uint32Array(64);
  private len = 0;
  poppedDist = 0;

  constructor(private readonly maxFirst = false) {}

  get size(): number {
    return this.len;
  }

  private less(i: number, j: number): boolean {
    return this.maxFirst ? this.dists[i] > this.dists[j] : this.dists[i] < this.dists[j];
  }

  private swap(i: number, j: number): void {
    const d = this.dists[i];
    this.dists[i] = this.dists[j];
    this.dists[j] = d;
    const h = this.handles[i];
    this.handles[i] = this.handles[j];
    this.handles[j] = h;
  }

  push(handle: number, dist: number): void {
    if (this.len === this.dists.length) {
      const d = new Float64Array(this.dists.length * 2);
      d.set(this.dists);
      this.dists = d;
      const n = new Uint32Array(this.handles.length * 2);
      n.set(this.handles);
      this.handles = n;
    }
    let i = this.len++;
    this.dists[i] = dist;
    this.handles[i] = handle;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p);
      i = p;
    }
  }

  pop(): number {
    const top = this.handles[0];
    this.poppedDist = this.dists[0];
    this.len--;
    if (this.len > 0) {
      this.dists[0] = this.dists[this.len];
      this.handles[0] = this.handles[this.len];
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.len && this.less(l, m)) m = l;
        if (r < this.len && this.less(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }

  peekDist(): number {
    return this.dists[0];
  }

  /** Evict everything (largest-first for a max-heap), sorted by distance ascending. */
  drain(): { h: number; d: number }[] {
    const out: { h: number; d: number }[] = [];
    while (this.len > 0) {
      const h = this.pop();
      out.push({ h, d: this.poppedDist });
    }
    out.sort((a, b) => a.d - b.d);
    return out;
  }
}

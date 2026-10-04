import { BufReader, BufWriter } from './binary.js';
import { makeDistance, prepareVector, type DistanceFn } from './distance.js';
import { MinHeap } from './heap.js';
import { hashSeed, mulberry32 } from './rng.js';
import type {
  IndexStats,
  Metric,
  SearchOptions,
  SearchResult,
  SearchTrace,
  TraceLayer,
} from './types.js';

const MAGIC = new Uint8Array([0x53, 0x4b, 0x56, 0x31]); // "SKV1"
const FORMAT_VERSION = 2;
const MAX_LEVEL = 32;

const METRIC_CODE: Record<Metric, number> = { euclidean: 0, cosine: 1, dot: 2 };
const METRIC_BY_CODE: Metric[] = ['euclidean', 'cosine', 'dot'];

export type Quantization = 'none' | 'sq8';
/**
 * per-node scalars derived from the u8 codes + calibration (never serialized —
 * recomputed on load): cosine/dot need `dotC`/`norm` for the symmetric
 * expansion; euclidean needs `selfW`/`xsm` for asymmetric distance computation.
 */
interface Aux {
  dotC: number;
  norm: number;
  selfW: number;
  xsm: number;
}

interface Node {
  id: string;
  vec: Float32Array | Uint8Array;
  level: number;
  deleted: boolean;
  /** neighbor handles per level; layer 0 is the dense one */
  links: number[][];
  aux: Aux | null;
}

interface Cand {
  h: number;
  d: number;
}

/** calibration: per-dimension affine map f32 → u8, frozen at `calibrate()` time */
export interface Calibration {
  min: Float32Array;
  max: Float32Array;
  /** step_d = (max_d − min_d) / 255 */
  step: Float32Array;
  /** c_d = step_d · min_d (cosine/dot expansion term) */
  c: Float32Array;
  /** Σ min_d² (cosine/dot expansion constant) */
  k: number;
}

/**
 * What the distance kernels see. Post-calibration the stored side is always
 * u8+aux. The *query* side depends on the metric: cosine/dot search in fully
 * quantized space (query quantized too); euclidean uses asymmetric distance
 * computation — the query keeps full precision (carrying `qs = v⊙step`,
 * `qm = Σ v·min`, `q2 = Σ v²`) and only the data side is dequantized.
 */
interface QVec {
  vec: Float32Array | Uint8Array;
  aux: Aux | null;
  qs?: Float32Array;
  qm?: number;
  q2?: number;
}

export interface HnswParams {
  dim: number;
  metric?: Metric;
  /** max connections per node on layers > 0; layer 0 uses 2×M */
  M?: number;
  efConstruction?: number;
  /** salt for layer-assignment RNG; same seed + same op order ⇒ identical graph */
  seed?: number;
  /** 'sq8' stores 1 byte per dimension after `calibrate()` (default 'none') */
  quantization?: Quantization;
}

export interface TracedSearch {
  results: SearchResult[];
  trace: SearchTrace;
}

/**
 * HNSW (Hierarchical Navigable Small World) index, implemented from scratch
 * after Malkov & Yashunin 2016: exponential layer assignment, greedy descent
 * through upper layers, ef-bounded beam search on layer 0, and diversity-aware
 * neighbor selection (Algorithm 4, keepPruned=false).
 *
 * Deletion is soft: deleted nodes stay as traversal anchors but are filtered
 * from results — the standard hnswlib tradeoff. Compaction happens via
 * `compacted()` / snapshot checkpointing.
 *
 * Scalar quantization: build with `quantization: 'sq8'`, insert freely in f32,
 * then `calibrate()` — vectors are rewritten as u8 with a per-dimension affine
 * map and all distances run in dequantizing integer space. 4× smaller vectors,
 * ~1-3% recall cost (see benchmarks).
 */
export class HnswIndex {
  readonly dim: number;
  readonly metric: Metric;
  readonly M: number;
  readonly M0: number;
  readonly efConstruction: number;
  readonly seed: number;
  readonly quantization: Quantization;

  private readonly mL: number;
  private nodes: Node[] = [];
  private idToHandle = new Map<string, number>();
  private entry = -1;
  private maxLevel = -1;
  private insertSeq = 0;
  private alive = 0;
  private deleted = 0;
  private calibration: Calibration | null = null;

  /** full-precision kernel (pre-calibration) */
  private distF32: DistanceFn;
  /** quantized symmetric kernel (post-calibration): u8 codes on both sides */
  private distQ: ((a: QVec, b: QVec) => number) | null = null;
  /** asymmetric kernel (calibrated euclidean only): exact query × dequantized code */
  private distAdc: ((node: QVec, q: QVec) => number) | null = null;

  constructor(params: HnswParams) {
    const { dim } = params;
    if (!Number.isInteger(dim) || dim < 1) throw new Error(`dim must be a positive integer, got ${dim}`);
    const metric = params.metric ?? 'cosine';
    const M = params.M ?? 16;
    const efConstruction = params.efConstruction ?? 200;
    const quantization = params.quantization ?? 'none';
    if (!Number.isInteger(M) || M < 2) throw new Error(`M must be an integer ≥ 2, got ${M}`);
    if (!Number.isInteger(efConstruction) || efConstruction < 1) {
      throw new Error(`efConstruction must be an integer ≥ 1, got ${efConstruction}`);
    }
    if (quantization !== 'none' && quantization !== 'sq8') {
      throw new Error(`quantization must be "none" or "sq8", got ${quantization}`);
    }
    this.dim = dim;
    this.metric = metric;
    this.M = M;
    this.M0 = 2 * M;
    this.efConstruction = efConstruction;
    this.seed = params.seed ?? 0x5356;
    this.quantization = quantization;
    this.mL = 1 / Math.log(M);
    this.distF32 = makeDistance(metric);
  }

  get size(): number {
    return this.alive;
  }

  get deletedCount(): number {
    return this.deleted;
  }

  get isCalibrated(): boolean {
    return this.calibration !== null;
  }

  /** bytes a stored vector occupies (excluding graph links) — one byte per dimension */
  get bytesPerVector(): number {
    return this.calibration !== null ? this.dim : this.dim * 4;
  }

  /**
   * Freeze per-dimension [min, max] over all stored vectors and rewrite every
   * vector as u8. One-shot: later inserts are quantized through the frozen
   * ranges (clamped) — the "train once, serve" model used by faiss/hnswlib.
   */
  calibrate(): void {
    if (this.quantization !== 'sq8') throw new Error('calibrate() requires quantization: "sq8"');
    if (this.calibration !== null) throw new Error('index is already calibrated');
    if (this.nodes.length === 0) throw new Error('cannot calibrate an empty index');

    const { dim } = this;
    const min = new Float32Array(dim).fill(Infinity);
    const max = new Float32Array(dim).fill(-Infinity);
    for (const node of this.nodes) {
      const v = node.vec as Float32Array;
      for (let d = 0; d < dim; d++) {
        const x = v[d]!;
        if (x < min[d]!) min[d] = x;
        if (x > max[d]!) max[d] = x;
      }
    }
    const step = new Float32Array(dim);
    const c = new Float32Array(dim);
    let k = 0;
    for (let d = 0; d < dim; d++) {
      if (!(max[d]! > min[d]!)) max[d] = min[d]! + 1e-6; // degenerate dimension
      step[d] = (max[d]! - min[d]!) / 255;
      c[d] = step[d]! * min[d]!;
      k += min[d]! * min[d]!;
    }
    this.calibration = { min, max, step, c, k };
    this.distQ = this.metric === 'euclidean' ? distQEuclidean(this.calibration) : distQIp(this.calibration, this.metric === 'cosine');
    this.distAdc = this.metric === 'euclidean' ? distQAdcEuclidean(this.calibration) : null;

    for (const node of this.nodes) {
      const u8 = this.quantizeToU8(node.vec as Float32Array);
      node.vec = u8;
      node.aux = this.auxOf(u8);
    }
  }

  // ------------------------------------------------------------------- write

  /** Insert or replace. Replacing soft-deletes the previous vector. */
  add(id: string, vec: ArrayLike<number>): void {
    if (this.calibration !== null && vec instanceof Uint8Array) {
      this.insertPrepared(id, vec, this.auxOf(vec));
      return;
    }
    const f = prepareVector(vec, this.metric, this.dim);
    if (this.calibration === null) {
      this.insertPrepared(id, f, null);
    } else {
      const u8 = this.quantizeToU8(f);
      this.insertPrepared(id, u8, this.auxOf(u8), f);
    }
  }

  private insertPrepared(id: string, vec: Float32Array | Uint8Array, aux: Aux | null, exact?: Float32Array): void {
    const old = this.idToHandle.get(id);
    if (old !== undefined) {
      const oldNode = this.nodes[old]!;
      if (!oldNode.deleted) {
        oldNode.deleted = true;
        this.alive--;
        this.deleted++;
      }
      this.idToHandle.delete(id);
    }

    const level = this.randomLevel(id);
    const h = this.nodes.length;
    this.nodes.push({
      id,
      vec,
      level,
      deleted: false,
      links: Array.from({ length: level + 1 }, () => [] as number[]),
      aux,
    });
    this.idToHandle.set(id, h);
    this.alive++;

    if (this.entry === -1) {
      this.entry = h;
      this.maxLevel = level;
      return;
    }

    const q: QVec =
      this.calibration !== null && this.metric === 'euclidean' && exact !== undefined
        ? this.adcQuery(exact)
        : { vec, aux };
    let eps: number[] = [this.entry];
    for (let l = this.maxLevel; l > level; l--) {
      eps = [this.greedy(q, eps[0]!, l)];
    }
    for (let l = Math.min(level, this.maxLevel); l >= 0; l--) {
      const { cands } = this.searchLayer(q, eps, this.efConstruction, l);
      const maxM = l === 0 ? this.M0 : this.M;
      const selected = this.selectNeighbors(cands, this.M);
      this.nodes[h]!.links[l] = selected;
      for (const nb of selected) {
        const nbLinks = this.nodes[nb]!.links[l]!;
        nbLinks.push(h);
        if (nbLinks.length > maxM) this.shrink(nb, l, maxM);
      }
      eps = cands.map((c) => c.h);
    }
    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entry = h;
    }
  }

  /** Soft-delete. Returns false when the id is absent or already deleted. */
  remove(id: string): boolean {
    const h = this.idToHandle.get(id);
    if (h === undefined) return false;
    const node = this.nodes[h]!;
    if (node.deleted) return false;
    node.deleted = true;
    this.idToHandle.delete(id);
    this.alive--;
    this.deleted++;
    return true;
  }

  // -------------------------------------------------------------------- read

  search(query: ArrayLike<number>, k = 10, options: SearchOptions = {}): SearchResult[] {
    return this.doSearch(query, k, options).results;
  }

  /** Same results as `search`, plus the full layer-by-layer traversal for visualization. */
  searchWithTrace(query: ArrayLike<number>, k = 10, options: SearchOptions = {}): TracedSearch {
    return this.doSearch(query, k, options, true);
  }

  private doSearch(query: ArrayLike<number>, k: number, options: SearchOptions, wantTrace = false): TracedSearch {
    const empty: TracedSearch = { results: [], trace: { layers: [], visitedTotal: 0 } };
    if (k < 1) return empty;
    if (this.entry === -1) return empty;
    const q = this.prepareQuery(query);
    const ef = Math.max(options.ef ?? Math.max(k, 16), k);
    const filter = options.filter;

    const trace: TraceLayer[] = wantTrace ? [] : [];
    let ep = this.entry;
    for (let l = this.maxLevel; l > 0; l--) {
      if (wantTrace) {
        const hops: { from: number; to: number; dist: number }[] = [];
        const start = ep;
        ep = this.greedy(q, ep, l, hops);
        trace.push({
          level: l,
          entry: this.nodes[start]!.id,
          hops: hops.map((x) => ({ from: this.nodes[x.from]!.id, to: this.nodes[x.to]!.id, dist: x.dist })),
          visited: hops.length + 1,
        });
      } else {
        ep = this.greedy(q, ep, l);
      }
    }

    const layer0Hops: { from: number; to: number; dist: number }[] | undefined = wantTrace ? [] : undefined;
    const { cands, visited } = this.searchLayer(q, [ep], ef, 0, layer0Hops, filter);
    const results: SearchResult[] = [];
    for (const c of cands) {
      const node = this.nodes[c.h]!;
      if (!node.deleted && (filter === undefined || filter(node.id))) {
        results.push({ id: node.id, dist: c.d });
        if (results.length === k) break;
      }
    }
    if (wantTrace) {
      trace.push({
        level: 0,
        entry: this.nodes[ep]!.id,
        hops: layer0Hops!.map((x) => ({ from: this.nodes[x.from]!.id, to: this.nodes[x.to]!.id, dist: x.dist })),
        visited,
      });
    }
    return {
      results,
      trace: { layers: trace, visitedTotal: trace.reduce((s, l) => s + l.visited, 0) },
    };
  }

  stats(): IndexStats {
    const levels: IndexStats['levels'] = [];
    for (let l = 0; l <= this.maxLevel; l++) {
      let nodes = 0;
      let degSum = 0;
      let maxDeg = 0;
      for (const node of this.nodes) {
        if (node.deleted || node.level < l) continue;
        nodes++;
        const deg = (node.links[l] ?? []).length;
        degSum += deg;
        if (deg > maxDeg) maxDeg = deg;
      }
      levels.push({ level: l, nodes, avgDegree: nodes > 0 ? degSum / nodes : 0, maxDegree: maxDeg });
    }
    return {
      count: this.alive,
      deleted: this.deleted,
      maxLevel: this.maxLevel,
      levels,
      params: { dim: this.dim, M: this.M, M0: this.M0, efConstruction: this.efConstruction, metric: this.metric },
    };
  }

  /** Per-level neighbor ids of a node — for visualization and introspection. */
  adjacency(id: string): string[][] | null {
    const h = this.idToHandle.get(id);
    if (h === undefined) return null;
    return this.nodes[h]!.links.map((layer) => layer.map((nb) => this.nodes[nb]!.id));
  }

  /**
   * Rebuild with only alive nodes, preserving insertion order. The result is a
   * fresh deterministic graph (levels are re-sampled, generation slots vanish)
   * — used by snapshot compaction to reclaim soft-deleted space. Calibration
   * carries over untouched.
   */
  compacted(): HnswIndex {
    const fresh = new HnswIndex({
      dim: this.dim,
      metric: this.metric,
      M: this.M,
      efConstruction: this.efConstruction,
      seed: this.seed,
      quantization: this.quantization,
    });
    if (this.calibration !== null) fresh.adoptCalibration(this.calibration);
    for (const node of this.nodes) {
      if (!node.deleted) {
        // calibrated euclidean: re-anchor inserts on the dequantized vector
        const exact = this.calibration !== null && this.metric === 'euclidean' ? this.dequantize(node.vec as Uint8Array) : undefined;
        fresh.insertPrepared(node.id, node.vec, node.aux, exact);
      }
    }
    return fresh;
  }

  // ------------------------------------------------------------------ kernel

  private quantizeToU8(v: Float32Array): Uint8Array {
    const cal = this.calibration!;
    const out = new Uint8Array(this.dim);
    for (let d = 0; d < this.dim; d++) {
      const x = v[d]!;
      const q = Math.round((x - cal.min[d]!) / cal.step[d]!);
      out[d] = q < 0 ? 0 : q > 255 ? 255 : q;
    }
    return out;
  }

  /** dequantized aux scalars, fully derivable from codes + calibration */
  private auxOf(u8: Uint8Array): Aux {
    const cal = this.calibration!;
    let dotC = 0;
    let norm2 = 0;
    let selfW = 0;
    let xsm = 0;
    for (let d = 0; d < this.dim; d++) {
      const q = u8[d]!;
      const v = q * cal.step[d]! + cal.min[d]!;
      dotC += q * cal.c[d]!;
      norm2 += v * v;
      selfW += q * q * cal.step[d]! * cal.step[d]!;
      xsm += q * cal.step[d]! * cal.min[d]!;
    }
    return { dotC, norm: Math.sqrt(norm2), selfW, xsm };
  }

  private needsAux(): boolean {
    return this.calibration !== null;
  }

  /** asymmetric query context: exact f32 query reduced to (v⊙step, v·min, ‖v‖²) */
  private adcQuery(f: Float32Array): QVec {
    const cal = this.calibration!;
    const qs = new Float32Array(this.dim);
    let qm = 0;
    let q2 = 0;
    for (let d = 0; d < this.dim; d++) {
      qs[d] = f[d]! * cal.step[d]!;
      qm += f[d]! * cal.min[d]!;
      q2 += f[d]! * f[d]!;
    }
    return { vec: f, aux: null, qs, qm, q2 };
  }

  private dequantize(u8: Uint8Array): Float32Array {
    const cal = this.calibration!;
    const out = new Float32Array(this.dim);
    for (let d = 0; d < this.dim; d++) out[d] = u8[d]! * cal.step[d]! + cal.min[d]!;
    return out;
  }

  /** f32 or u8 query, with aux scalars for the quantized cosine kernel */
  private prepareQuery(query: ArrayLike<number>): QVec {
    if (this.calibration === null) {
      if (query instanceof Uint8Array) throw new Error('internal: u8 query on uncalibrated index');
      return { vec: prepareVector(query, this.metric, this.dim), aux: null };
    }
    const f = prepareVector(query, this.metric, this.dim);
    if (this.metric === 'euclidean') return this.adcQuery(f);
    const u8 = this.quantizeToU8(f);
    return { vec: u8, aux: this.auxOf(u8) };
  }

  private dist(h: number, q: QVec): number {
    const node = this.nodes[h]!;
    if (this.calibration === null) return this.distF32(node.vec as Float32Array, q.vec as Float32Array);
    if (this.metric === 'euclidean') return this.distAdc!(node, q);
    return this.distQ!({ vec: node.vec, aux: node.aux }, q);
  }

  private distNN(a: number, b: number): number {
    if (this.calibration === null) {
      return this.distF32(this.nodes[a]!.vec as Float32Array, this.nodes[b]!.vec as Float32Array);
    }
    const na = this.nodes[a]!;
    const nb = this.nodes[b]!;
    return this.distQ!({ vec: na.vec, aux: na.aux }, { vec: nb.vec, aux: nb.aux });
  }

  // ------------------------------------------------------------------- graph

  private randomLevel(id: string): number {
    const rng = mulberry32(hashSeed(id, (this.seed ^ Math.imul(this.insertSeq, 0x9e3779b1)) >>> 0));
    this.insertSeq++;
    const u = rng();
    return Math.min(Math.floor(-Math.log(1 - u) * this.mL), MAX_LEVEL);
  }

  /** ef=1 greedy walk on one layer; records every edge it inspects when `hops` is given. */
  private greedy(q: QVec, ep: number, level: number, hops?: { from: number; to: number; dist: number }[]): number {
    let cur = ep;
    let curDist = this.dist(cur, q);
    for (;;) {
      let improved = false;
      for (const nb of this.nodes[cur]!.links[level] ?? []) {
        const d = this.dist(nb, q);
        hops?.push({ from: cur, to: nb, dist: d });
        if (d < curDist) {
          cur = nb;
          curDist = d;
          improved = true;
        }
      }
      if (!improved) break;
    }
    return cur;
  }

  /**
   * ef-bounded beam search on one layer. Returns alive+dead candidates, sorted near→far.
   * When `filter` is set, filtered nodes are still expanded but never enter the
   * result beam — the ef budget is spent on admissible nodes only.
   */
  private searchLayer(
    q: QVec,
    eps: number[],
    ef: number,
    level: number,
    hops?: { from: number; to: number; dist: number }[],
    filter?: (id: string) => boolean,
  ): { cands: Cand[]; visited: number } {
    const visited = new Set<number>(eps);
    const frontier = new MinHeap<Cand>((a, b) => a.d < b.d);
    const best = new MinHeap<Cand>((a, b) => a.d > b.d); // max-heap on distance
    for (const e of eps) {
      const d = this.dist(e, q);
      frontier.push({ h: e, d });
      if ((filter === undefined || filter(this.nodes[e]!.id)) && !this.nodes[e]!.deleted) {
        best.push({ h: e, d });
      }
    }
    while (frontier.size > 0) {
      const c = frontier.pop()!;
      const worst = best.peek();
      if (worst !== undefined && c.d > worst.d && best.size >= ef) break;
      for (const nb of this.nodes[c.h]!.links[level] ?? []) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const d = this.dist(nb, q);
        hops?.push({ from: c.h, to: nb, dist: d });
        const admissible = (filter === undefined || filter(this.nodes[nb]!.id)) && !this.nodes[nb]!.deleted;
        if (admissible && (best.size < ef || d < best.peek()!.d)) {
          frontier.push({ h: nb, d });
          best.push({ h: nb, d });
          if (best.size > ef) best.pop();
        } else if (!admissible) {
          frontier.push({ h: nb, d });
        }
      }
    }
    const cands: Cand[] = [];
    while (best.size > 0) cands.push(best.pop()!);
    cands.sort((a, b) => a.d - b.d);
    return { cands, visited: visited.size };
  }

  /**
   * Diversity-aware neighbor selection (paper Algorithm 4, keepPruned=false):
   * scan candidates near→far, keep one only if it is closer to the base point
   * than to every already-kept neighbor. Prevents mutual-cluster hubs.
   */
  private selectNeighbors(cands: Cand[], M: number): number[] {
    const sorted = cands.slice().sort((a, b) => a.d - b.d);
    const kept: Cand[] = [];
    for (const c of sorted) {
      if (kept.length >= M) break;
      let ok = true;
      for (const r of kept) {
        if (this.distNN(c.h, r.h) < c.d) {
          ok = false;
          break;
        }
      }
      if (ok) kept.push(c);
    }
    return kept.map((c) => c.h);
  }

  /** Re-select a node's neighbor list after a new edge pushed it past `maxM`. */
  private shrink(h: number, level: number, maxM: number): void {
    const node = this.nodes[h]!;
    const links = node.links[level]!;
    const cands = links.map((x) => ({ h: x, d: this.distNN(x, h) }));
    node.links[level] = this.selectNeighbors(cands, maxM);
  }

  // ------------------------------------------------------------- calibration IO

  /** adopt an already-computed calibration (compaction path); rewrites nothing */
  private adoptCalibration(cal: Calibration): void {
    if (this.quantization !== 'sq8') throw new Error('adoptCalibration requires quantization: "sq8"');
    if (this.calibration !== null) throw new Error('index is already calibrated');
    if (cal.min.length !== this.dim) throw new Error('calibration dim mismatch');
    this.calibration = cal;
    this.distQ = this.metric === 'euclidean' ? distQEuclidean(cal) : distQIp(cal, this.metric === 'cosine');
    this.distAdc = this.metric === 'euclidean' ? distQAdcEuclidean(cal) : null;
  }

  // ------------------------------------------------------------- persistence

  serialize(): Uint8Array {
    const w = new BufWriter(1 << 16);
    w.bytes(MAGIC);
    w.u32(FORMAT_VERSION);
    w.u32(this.dim);
    w.u8(METRIC_CODE[this.metric]);
    w.u32(this.M);
    w.u32(this.efConstruction);
    w.u32(this.seed);
    const cal = this.calibration;
    w.u8(cal !== null ? 2 : this.quantization === 'sq8' ? 1 : 0); // 0 f32 · 1 sq8-uncalibrated · 2 sq8-calibrated
    if (cal !== null) {
      for (let d = 0; d < this.dim; d++) w.f32(cal.min[d]!).f32(cal.max[d]!);
    }
    w.u32(this.nodes.length);
    w.i32(this.entry);
    w.u32(this.maxLevel < 0 ? 0 : this.maxLevel);
    for (const node of this.nodes) {
      w.str(node.id);
      w.u8(node.level);
      w.u8(node.deleted ? 1 : 0);
      if (cal !== null) {
        const v = node.vec as Uint8Array;
        for (let i = 0; i < this.dim; i++) w.u8(v[i]!);
      } else {
        const v = node.vec as Float32Array;
        for (let i = 0; i < this.dim; i++) w.f32(v[i]!);
      }
      for (let l = 0; l <= node.level; l++) {
        const links = node.links[l] ?? [];
        w.u16(links.length);
        for (const nb of links) w.u32(nb);
      }
    }
    return w.finish();
  }

  static deserialize(data: Uint8Array): HnswIndex {
    const r = new BufReader(data);
    for (let i = 0; i < MAGIC.length; i++) {
      if (r.u8() !== MAGIC[i]) throw new Error('not a skipverse index (bad magic)');
    }
    const version = r.u32();
    if (version !== FORMAT_VERSION && version !== 1) throw new Error(`unsupported index format version ${version}`);
    const dim = r.u32();
    const metric = METRIC_BY_CODE[r.u8()];
    if (metric === undefined) throw new Error('bad metric code in index');
    const M = r.u32();
    const efConstruction = r.u32();
    const seed = r.u32();
    const quantCode = version >= 2 ? r.u8() : 0;
    const idx = new HnswIndex({
      dim,
      metric,
      M,
      efConstruction,
      seed,
      quantization: quantCode === 0 ? 'none' : 'sq8',
    });
    if (quantCode === 2) {
      const min = new Float32Array(dim);
      const max = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        min[d] = r.f32();
        max[d] = r.f32();
      }
      const step = new Float32Array(dim);
      const c = new Float32Array(dim);
      let k = 0;
      for (let d = 0; d < dim; d++) {
        step[d] = (max[d]! - min[d]!) / 255;
        c[d] = step[d]! * min[d]!;
        k += min[d]! * min[d]!;
      }
      idx.calibration = { min, max, step, c, k };
      idx.distQ = metric === 'euclidean' ? distQEuclidean(idx.calibration) : distQIp(idx.calibration, metric === 'cosine');
    }
    const count = r.u32();
    const entry = r.i32();
    const maxLevelRaw = r.u32();
    for (let i = 0; i < count; i++) {
      const id = r.str();
      const level = r.u8();
      const deleted = r.u8() === 1;
      if (level > MAX_LEVEL) throw new Error(`corrupt index: node level ${level} > ${MAX_LEVEL}`);
      let vec: Float32Array | Uint8Array;
      if (quantCode === 2) {
        const u8 = new Uint8Array(dim);
        for (let j = 0; j < dim; j++) u8[j] = r.u8();
        vec = u8;
      } else {
        const f = new Float32Array(dim);
        for (let j = 0; j < dim; j++) f[j] = r.f32();
        vec = f;
      }
      const links: number[][] = [];
      for (let l = 0; l <= level; l++) {
        const deg = r.u16();
        const layerLinks: number[] = [];
        for (let j = 0; j < deg; j++) {
          const nb = r.u32();
          if (nb >= count) throw new Error(`corrupt index: neighbor handle ${nb} out of range`);
          layerLinks.push(nb);
        }
        links.push(layerLinks);
      }
      idx.nodes.push({ id, vec, level, deleted, links, aux: null });
      if (!deleted) {
        idx.idToHandle.set(id, i);
        idx.alive++;
      } else {
        idx.deleted++;
      }
    }
    // aux scalars are pure functions of (codes, calibration) — rebuild, don't store
    if (quantCode === 2) {
      for (const node of idx.nodes) node.aux = idx.auxOf(node.vec as Uint8Array);
    }
    if (entry !== -1) {
      if (entry >= count) throw new Error('corrupt index: entry point out of range');
      // a deleted entry point is legal: it still serves as a traversal anchor
      if (idx.nodes[entry]!.level !== maxLevelRaw) throw new Error('corrupt index: maxLevel inconsistent with entry point');
      idx.entry = entry;
      idx.maxLevel = maxLevelRaw;
    }
    idx.insertSeq = count;
    return idx;
  }
}

// ------------------------------------------------------------ quantized kernels

/** euclidean over u8 codes on both sides (node-node distances in ADC mode) */
function distQEuclidean(cal: Calibration): (a: QVec, b: QVec) => number {
  const { step, min } = cal;
  const dim = min.length;
  return (a, b) => {
    const va = a.vec as Uint8Array;
    const vb = b.vec as Uint8Array;
    let s = 0;
    for (let d = 0; d < dim; d++) {
      const diff = va[d]! - vb[d]!;
      s += diff * diff * step[d]!;
    }
    return Math.sqrt(s);
  };
}

/**
 * Asymmetric distance computation for euclidean: the query keeps full f32
 * precision and the data side is dequantized in-register.
 *   ‖v_a − v_b‖² = ‖v_a‖² − 2·v_a·v_b + ‖v_b‖²
 *   v_a·v_b      = dot(qs, q_b) + qm          (qs = v_a⊙step, qm = Σ v_a·min)
 *   ‖v_b‖²       = selfW + 2·xsm + K          (per-node precomputed scalars)
 * so each pair costs one dim-loop plus three scalars, and the query's
 * quantization noise — half of the total in symmetric codes — disappears.
 */
function distQAdcEuclidean(cal: Calibration): (node: QVec, q: QVec) => number {
  const { step, min, k } = cal;
  const dim = min.length;
  return (node, q) => {
    const qb = node.vec as Uint8Array;
    const qs = q.qs!;
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += qs[d]! * qb[d]!;
    const d2 = q.q2! - 2 * (dot + q.qm!) + node.aux!.selfW + 2 * node.aux!.xsm + k;
    return Math.sqrt(Math.max(0, d2));
  };
}

/**
 * cosine / dot over u8 codes on both sides. With v = q·step + min:
 *   v·v′ = q·W·q′ + c·q + c·q′ + K   (W = step²ᵈ, c = step·min, K = Σ min²)
 * Both sides carry `dotC = Σ c_d·q_d` and the dequantized L2 norm as
 * precomputed scalars, so the per-pair cost is one weighted dot plus three
 * scalars — no per-dimension affine work.
 */
function distQIp(cal: Calibration, cosine: boolean): (a: QVec, b: QVec) => number {
  const { step, k, min } = cal;
  const dim = min.length;
  const wDot = (a: Uint8Array, b: Uint8Array): number => {
    let s = 0;
    for (let d = 0; d < dim; d++) s += a[d]! * b[d]! * step[d]! * step[d]!;
    return s;
  };
  return (a, b) => {
    const ip = wDot(a.vec as Uint8Array, b.vec as Uint8Array) + a.aux!.dotC + b.aux!.dotC + k;
    if (!cosine) return -ip;
    const denom = a.aux!.norm * b.aux!.norm;
    return denom === 0 ? 1 : 1 - ip / denom;
  };
}

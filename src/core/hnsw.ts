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
const FORMAT_VERSION = 1;
const MAX_LEVEL = 32;

const METRIC_CODE: Record<Metric, number> = { euclidean: 0, cosine: 1, dot: 2 };
const METRIC_BY_CODE: Metric[] = ['euclidean', 'cosine', 'dot'];

interface Node {
  id: string;
  vec: Float32Array;
  level: number;
  deleted: boolean;
  /** neighbor handles per level; layer 0 is the dense one */
  links: number[][];
}

interface Cand {
  h: number;
  d: number;
}

export interface HnswParams {
  dim: number;
  metric?: Metric;
  /** max connections per node on layers > 0; layer 0 uses 2×M */
  M?: number;
  efConstruction?: number;
  /** salt for layer-assignment RNG; same seed + same op order ⇒ identical graph */
  seed?: number;
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
 * snapshot checkpointing.
 */
export class HnswIndex {
  readonly dim: number;
  readonly metric: Metric;
  readonly M: number;
  readonly M0: number;
  readonly efConstruction: number;
  readonly seed: number;

  private readonly mL: number;
  private readonly distFn: DistanceFn;
  private nodes: Node[] = [];
  private idToHandle = new Map<string, number>();
  private entry = -1;
  private maxLevel = -1;
  private insertSeq = 0;
  private alive = 0;
  private deleted = 0;

  constructor(params: HnswParams) {
    const { dim } = params;
    if (!Number.isInteger(dim) || dim < 1) throw new Error(`dim must be a positive integer, got ${dim}`);
    const metric = params.metric ?? 'cosine';
    const M = params.M ?? 16;
    const efConstruction = params.efConstruction ?? 200;
    if (!Number.isInteger(M) || M < 2) throw new Error(`M must be an integer ≥ 2, got ${M}`);
    if (!Number.isInteger(efConstruction) || efConstruction < 1) {
      throw new Error(`efConstruction must be an integer ≥ 1, got ${efConstruction}`);
    }
    this.dim = dim;
    this.metric = metric;
    this.M = M;
    this.M0 = 2 * M;
    this.efConstruction = efConstruction;
    this.seed = params.seed ?? 0x5356;
    this.mL = 1 / Math.log(M);
    this.distFn = makeDistance(metric);
  }

  get size(): number {
    return this.alive;
  }

  get deletedCount(): number {
    return this.deleted;
  }

  /** Insert or replace. Replacing soft-deletes the previous vector. */
  add(id: string, vec: ArrayLike<number>): void {
    const v = prepareVector(vec, this.metric, this.dim);
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
      vec: v,
      level,
      deleted: false,
      links: Array.from({ length: level + 1 }, () => [] as number[]),
    });
    this.idToHandle.set(id, h);
    this.alive++;

    if (this.entry === -1) {
      this.entry = h;
      this.maxLevel = level;
      return;
    }

    let eps: number[] = [this.entry];
    for (let l = this.maxLevel; l > level; l--) {
      eps = [this.greedy(v, eps[0]!, l)];
    }
    for (let l = Math.min(level, this.maxLevel); l >= 0; l--) {
      const { cands } = this.searchLayer(v, eps, this.efConstruction, l);
      const maxM = l === 0 ? this.M0 : this.M;
      const selected = this.selectNeighbors(v, cands, this.M);
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
    const q = prepareVector(query, this.metric, this.dim);
    const ef = Math.max(options.ef ?? Math.max(k, 16), k);

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
    const { cands, visited } = this.searchLayer(q, [ep], ef, 0, layer0Hops);
    const results: SearchResult[] = [];
    for (const c of cands) {
      if (!this.nodes[c.h]!.deleted) {
        results.push({ id: this.nodes[c.h]!.id, dist: c.d });
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

  // ------------------------------------------------------------------ graph

  private distTo(h: number, q: Float32Array): number {
    return this.distFn(this.nodes[h]!.vec, q);
  }

  private randomLevel(id: string): number {
    const rng = mulberry32(hashSeed(id, (this.seed ^ Math.imul(this.insertSeq, 0x9e3779b1)) >>> 0));
    this.insertSeq++;
    const u = rng();
    return Math.min(Math.floor(-Math.log(1 - u) * this.mL), MAX_LEVEL);
  }

  /** ef=1 greedy walk on one layer; records every edge it inspects when `hops` is given. */
  private greedy(q: Float32Array, ep: number, level: number, hops?: { from: number; to: number; dist: number }[]): number {
    let cur = ep;
    let curDist = this.distTo(cur, q);
    for (;;) {
      let improved = false;
      for (const nb of this.nodes[cur]!.links[level] ?? []) {
        const d = this.distTo(nb, q);
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

  /** ef-bounded beam search on one layer. Returns alive+dead candidates, sorted near→far. */
  private searchLayer(
    q: Float32Array,
    eps: number[],
    ef: number,
    level: number,
    hops?: { from: number; to: number; dist: number }[],
  ): { cands: Cand[]; visited: number } {
    const visited = new Set<number>(eps);
    const frontier = new MinHeap<Cand>((a, b) => a.d < b.d);
    const best = new MinHeap<Cand>((a, b) => a.d > b.d); // max-heap on distance
    for (const e of eps) {
      const d = this.distTo(e, q);
      frontier.push({ h: e, d });
      best.push({ h: e, d });
    }
    while (frontier.size > 0) {
      const c = frontier.pop()!;
      const worst = best.peek()!;
      if (c.d > worst.d && best.size >= ef) break;
      for (const nb of this.nodes[c.h]!.links[level] ?? []) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const d = this.distTo(nb, q);
        hops?.push({ from: c.h, to: nb, dist: d });
        if (best.size < ef || d < best.peek()!.d) {
          frontier.push({ h: nb, d });
          best.push({ h: nb, d });
          if (best.size > ef) best.pop();
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
  private selectNeighbors(base: Float32Array, cands: Cand[], M: number): number[] {
    const sorted = cands.slice().sort((a, b) => a.d - b.d);
    const kept: Cand[] = [];
    for (const c of sorted) {
      if (kept.length >= M) break;
      const cv = this.nodes[c.h]!.vec;
      let ok = true;
      for (const r of kept) {
        if (this.distFn(cv, this.nodes[r.h]!.vec) < c.d) {
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
    const cands = links.map((x) => ({ h: x, d: this.distTo(x, node.vec) }));
    node.links[level] = this.selectNeighbors(node.vec, cands, maxM);
  }

  /** Per-level neighbor ids of a node — for visualization and introspection. */
  adjacency(id: string): string[][] | null {
    const h = this.idToHandle.get(id);
    if (h === undefined) return null;
    return this.nodes[h]!.links.map((layer) => layer.map((nb) => this.nodes[nb]!.id));
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
    w.u32(this.nodes.length);
    w.i32(this.entry);
    w.u32(this.maxLevel < 0 ? 0 : this.maxLevel);
    for (const node of this.nodes) {
      w.str(node.id);
      w.u8(node.level);
      w.u8(node.deleted ? 1 : 0);
      for (let i = 0; i < this.dim; i++) w.f32(node.vec[i]!);
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
    if (version !== FORMAT_VERSION) throw new Error(`unsupported index format version ${version}`);
    const dim = r.u32();
    const metric = METRIC_BY_CODE[r.u8()];
    if (metric === undefined) throw new Error('bad metric code in index');
    const M = r.u32();
    const efConstruction = r.u32();
    const seed = r.u32();
    const count = r.u32();
    const entry = r.i32();
    const maxLevelRaw = r.u32();
    const idx = new HnswIndex({ dim, metric, M, efConstruction, seed });
    for (let i = 0; i < count; i++) {
      const id = r.str();
      const level = r.u8();
      const deleted = r.u8() === 1;
      if (level > MAX_LEVEL) throw new Error(`corrupt index: node level ${level} > ${MAX_LEVEL}`);
      const vec = new Float32Array(dim);
      for (let j = 0; j < dim; j++) vec[j] = r.f32();
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
      idx.nodes.push({ id, vec, level, deleted, links });
      if (!deleted) {
        idx.idToHandle.set(id, i);
        idx.alive++;
      } else {
        idx.deleted++;
      }
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

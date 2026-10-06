import * as fs from 'node:fs';
import * as path from 'node:path';
import { HnswIndex, type TracedSearch } from '../core/hnsw.js';
import { prepareVector } from '../core/distance.js';
import type { IndexStats, Metric, SearchOptions, SearchResult } from '../core/types.js';
import { appendWal, appendWalBatch, loadWal } from './wal.js';

const META_FILE = 'meta.json';
const SNAPSHOT_FILE = 'snapshot.bin';
const SNAPSHOT_TMP = 'snapshot.bin.tmp';
const WAL_FILE = 'wal.log';
const SNAPSHOT_HEADER = 4; // u32 lastSeq

export interface StoreOptions {
  dataDir: string;
  dim: number;
  metric?: Metric;
  M?: number;
  efConstruction?: number;
  seed?: number;
  /** scalar quantization for the stored index: 'sq8' (4× smaller) or 'sq4' (8×, lossy) */
  quantization?: 'sq8' | 'sq4';
  /** auto-checkpoint after this many WAL ops; 0 disables (default 4096) */
  checkpointEvery?: number;
  /**
   * fsync the WAL after every append and the snapshot after every checkpoint
   * (default false). Real crash durability at a throughput cost; without it a
   * power loss may lose recent ops even though process death does not.
   */
  fsync?: boolean;
}

export interface StoreInfo extends IndexStats {
  dataDir: string;
  walBytes: number;
  walOpsSinceCheckpoint: number;
}

interface Meta {
  version: 1;
  dim: number;
  metric: Metric;
  M: number;
  efConstruction: number;
  seed: number;
  quantization: 'none' | 'sq8' | 'sq4';
}

/**
 * Durable vector store = HNSW index + CRC-framed WAL + atomic snapshot
 * checkpoints. Open path: load snapshot → replay WAL ops newer than the
 * snapshot's seq → truncate any torn WAL tail. All IO is synchronous by
 * design: operations are small, ordering is deterministic, and the code
 * stays honest about crash boundaries.
 */
export class VectorStore {
  /** Swapped out by `compact()`; treat as immutable between calls. */
  index: HnswIndex;
  readonly dataDir: string;

  private walFd: number;
  private walSeq: number;
  private opsSinceCheckpoint = 0;
  private readonly checkpointEvery: number;
  private readonly fsync: boolean;
  private readonly walPath: string;
  private readonly snapshotPath: string;
  private readonly lockPath: string;

  private constructor(opts: StoreOptions, meta: Meta) {
    this.dataDir = opts.dataDir;
    this.checkpointEvery = opts.checkpointEvery ?? 4096;
    this.fsync = opts.fsync ?? false;
    this.lockPath = path.join(opts.dataDir, 'lock');
    this.index = new HnswIndex({
      dim: meta.dim,
      metric: meta.metric,
      M: meta.M,
      efConstruction: meta.efConstruction,
      seed: meta.seed,
      quantization: meta.quantization,
    });
    this.walPath = path.join(opts.dataDir, WAL_FILE);
    this.snapshotPath = path.join(opts.dataDir, SNAPSHOT_FILE);
    let lastSeq = 0;

    if (fs.existsSync(this.snapshotPath)) {
      const raw = fs.readFileSync(this.snapshotPath);
      if (raw.length < SNAPSHOT_HEADER) throw new Error(`corrupt snapshot: ${raw.length} bytes`);
      const seqView = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      lastSeq = seqView.getUint32(0, true);
      this.index = HnswIndex.deserialize(raw.subarray(SNAPSHOT_HEADER));
    }

    let torn = false;
    if (fs.existsSync(this.walPath)) {
      const { ops, torn: t } = loadWal(this.walPath, meta.dim);
      torn = t;
      for (const op of ops) {
        if (op.seq <= lastSeq) continue;
        if (op.kind === 'upsert') this.index.add(op.id, op.vec);
        else this.index.remove(op.id);
        lastSeq = Math.max(lastSeq, op.seq);
      }
    }
    this.walSeq = lastSeq;
    if (torn) {
      console.warn(`[skipverse] WAL had a torn tail; recovered up to byte boundary`);
    }
    this.walFd = fs.openSync(this.walPath, 'a');
    this.acquireLock();
  }

  /** Advisory single-writer lock. A lock from a dead process is auto-cleared. */
  private acquireLock(): void {
    if (fs.existsSync(this.lockPath)) {
      let holder: { pid?: number } = {};
      try {
        holder = JSON.parse(fs.readFileSync(this.lockPath, 'utf8'));
      } catch {
        // unreadable lock — treat as stale
      }
      const pid = holder.pid ?? -1;
      let alive = false;
      try {
        process.kill(pid, 0);
        alive = true;
      } catch {
        alive = pid === process.pid; // EPERM means "exists"; ESRCH means "gone"
      }
      if (alive && pid !== process.pid) {
        throw new Error(`store is locked by another process (pid ${pid}); if this is wrong, delete ${this.lockPath}`);
      }
      console.warn(`[skipverse] clearing stale lock from pid ${pid}`);
    }
    fs.writeFileSync(this.lockPath, JSON.stringify({ pid: process.pid, at: new Date().toISOString() }));
  }

  private releaseLock(): void {
    try {
      fs.unlinkSync(this.lockPath);
    } catch {
      // already gone
    }
  }

  /** Open (or create) a store. `dim` must match any existing meta.json. */
  static open(opts: StoreOptions): VectorStore {
    fs.mkdirSync(opts.dataDir, { recursive: true });
    const metaPath = path.join(opts.dataDir, META_FILE);
    let meta: Meta;
    if (fs.existsSync(metaPath)) {
      meta = JSON.parse(fs.readFileSync(metaPath, 'utf8')) as Meta;
      if (meta.version !== 1) throw new Error(`unsupported meta version ${meta.version}`);
      if (opts.dim !== meta.dim) {
        throw new Error(`dim mismatch: store has ${meta.dim}, requested ${opts.dim}`);
      }
      const conflicts: string[] = [];
      if (opts.metric && opts.metric !== meta.metric) conflicts.push(`metric ${meta.metric} → ${opts.metric}`);
      if (opts.quantization && opts.quantization !== meta.quantization) {
        conflicts.push(`quantization ${meta.quantization} → ${opts.quantization}`);
      }
      if (opts.M && opts.M !== meta.M) conflicts.push(`M ${meta.M} → ${opts.M}`);
      if (opts.efConstruction && opts.efConstruction !== meta.efConstruction) {
        conflicts.push(`efConstruction ${meta.efConstruction} → ${opts.efConstruction}`);
      }
      if (conflicts.length > 0) throw new Error(`store config conflict: ${conflicts.join('; ')}`);
    } else {
      meta = {
        version: 1,
        dim: opts.dim,
        metric: opts.metric ?? 'cosine',
        M: opts.M ?? 16,
        efConstruction: opts.efConstruction ?? 200,
        seed: opts.seed ?? 0x5356,
        quantization: opts.quantization ?? 'none',
      };
      fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
    }
    return new VectorStore(opts, meta);
  }

  upsert(id: string, vec: ArrayLike<number>): void {
    const v = prepareVector(vec, this.index.metric, this.index.dim);
    this.index.add(id, v);
    this.appendOp({ seq: ++this.walSeq, kind: 'upsert', id, vec: v });
  }

  upsertBatch(entries: { id: string; vec: ArrayLike<number> }[]): void {
    if (entries.length === 0) return;
    // validate everything first so a bad vector can't half-apply the batch
    const prepared = entries.map((e) => ({
      id: e.id,
      vec: prepareVector(e.vec, this.index.metric, this.index.dim),
    }));
    const ops: Parameters<typeof appendWal>[1][] = [];
    for (const { id, vec } of prepared) {
      this.index.add(id, vec);
      this.walSeq++;
      ops.push({ seq: this.walSeq, kind: 'upsert' as const, id, vec });
    }
    appendWalBatch(this.walFd, ops, this.index.dim);
    if (this.fsync) fs.fsyncSync(this.walFd);
    this.opsSinceCheckpoint += ops.length;
    if (this.checkpointEvery > 0 && this.opsSinceCheckpoint >= this.checkpointEvery) {
      this.checkpoint();
    }
  }

  remove(id: string): boolean {
    const ok = this.index.remove(id);
    if (ok) this.appendOp({ seq: ++this.walSeq, kind: 'delete', id });
    return ok;
  }

  /** The stored (post-calibration: dequantized) vector for an id, or null. */
  get(id: string): Float32Array | null {
    return this.index.vector(id);
  }

  search(query: ArrayLike<number>, k = 10, options: SearchOptions = {}): SearchResult[] {
    return this.index.search(query, k, options);
  }

  searchWithTrace(query: ArrayLike<number>, k = 10, options: SearchOptions = {}): TracedSearch {
    return this.index.searchWithTrace(query, k, options);
  }

  info(): StoreInfo {
    return {
      ...this.index.stats(),
      dataDir: this.dataDir,
      walBytes: this.opsSinceCheckpoint === 0 ? 0 : fs.statSync(this.walPath).size,
      walOpsSinceCheckpoint: this.opsSinceCheckpoint,
    };
  }

  /** Calibrate the underlying quantized index (see HnswIndex.calibrate) and rotate a snapshot. */
  calibrate(): void {
    this.index.calibrate();
    this.checkpoint();
  }

  /** Write a snapshot (atomic rename), then truncate the WAL. */
  checkpoint(): void {
    const body = this.index.serialize();
    const out = Buffer.allocUnsafe(SNAPSHOT_HEADER + body.length);
    new DataView(out.buffer, out.byteOffset, out.byteLength).setUint32(0, this.walSeq, true);
    out.set(body, SNAPSHOT_HEADER);
    const tmp = path.join(this.dataDir, SNAPSHOT_TMP);
    fs.writeFileSync(tmp, out);
    if (this.fsync) {
      const fd = fs.openSync(tmp, 'r+'); // FlushFileBuffers needs a write handle on Windows
      fs.fsyncSync(fd);
      fs.closeSync(fd);
    }
    fs.renameSync(tmp, this.snapshotPath);
    fs.truncateSync(this.walPath, 0);
    this.opsSinceCheckpoint = 0;
  }

  /**
   * Rebuild the index from alive vectors only (drops soft-deleted slots) and
   * rotate a fresh snapshot + empty WAL. Deterministic; levels are re-sampled.
   */
  compact(): { before: number; after: number } {
    const before = this.index.size + this.index.deletedCount;
    this.index = this.index.compacted();
    this.checkpoint();
    return { before, after: this.index.size };
  }

  close(checkpoint = false): void {
    if (checkpoint) this.checkpoint();
    if (this.walFd === -1) return;
    fs.closeSync(this.walFd);
    this.walFd = -1;
    this.releaseLock();
  }

  private appendOp(op: Parameters<typeof appendWal>[1]): void {
    appendWal(this.walFd, op, this.index.dim);
    if (this.fsync) fs.fsyncSync(this.walFd);
    this.opsSinceCheckpoint++;
    if (this.checkpointEvery > 0 && this.opsSinceCheckpoint >= this.checkpointEvery) {
      this.checkpoint();
    }
  }
}

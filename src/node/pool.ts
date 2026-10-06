import { availableParallelism } from 'node:os';
import { pathToFileURL } from 'node:url';
import { type MessagePort, isMainThread, parentPort, workerData, Worker } from 'node:worker_threads';
import { HnswIndex } from '../core/hnsw.js';
import type { SearchResult } from '../core/types.js';
import type { VectorStore } from './store.js';

/**
 * Concurrent read pool: N worker threads each hold a read-only snapshot of the
 * store's index and serve `search` off the main thread, so a search storm
 * never blocks writes (and vice versa).
 *
 * Snapshot semantics: workers start from the index as of pool creation. The
 * main thread does NOT auto-sync after writes — call {@link SearchPool.refresh}
 * after upsert/remove for new data to become visible to the readers.
 *
 * The worker branch lives in this same file (the module re-enters itself as
 * the worker entry point), which keeps the implementation a single
 * zero-dependency file that works both from TS source (tsx) and compiled JS.
 */

type WorkerInbound =
  | { type: 'search'; reqId: number; vec: number[]; k: number; ef?: number }
  | { type: 'load'; data: Uint8Array }
  | { type: 'close' };

type WorkerOutbound =
  | { type: 'result'; reqId: number; results: SearchResult[] }
  | { type: 'error'; reqId?: number; message: string };

/** Worker-side loop: hold one deserialized index and answer search requests. */
function runWorker(port: MessagePort, init: { data: Uint8Array }): void {
  let index = HnswIndex.deserialize(init.data);
  port.on('message', (msg: WorkerInbound) => {
    try {
      if (msg.type === 'load') {
        index = HnswIndex.deserialize(msg.data);
      } else if (msg.type === 'search') {
        const results = index.search(msg.vec, msg.k, { ef: msg.ef });
        port.postMessage({ type: 'result', reqId: msg.reqId, results } satisfies WorkerOutbound);
      } else {
        port.close();
      }
    } catch (err) {
      port.postMessage({
        type: 'error',
        reqId: msg.type === 'search' ? msg.reqId : undefined,
        message: err instanceof Error ? err.message : String(err),
      } satisfies WorkerOutbound);
    }
  });
}

if (!isMainThread) {
  const port = parentPort;
  if (!port) throw new Error('[skipverse] pool worker started without a parent port');
  runWorker(port, workerData as { data: Uint8Array });
}

export interface SearchPoolOptions {
  store: VectorStore;
  /** worker thread count (default: max(1, cpus-1), capped at 4) */
  workers?: number;
}

interface PendingSearch {
  resolve: (results: SearchResult[]) => void;
  reject: (err: Error) => void;
  worker: Worker;
}

export class SearchPool {
  private readonly store: VectorStore;
  private workers: Worker[] = [];
  private readonly pending = new Map<number, PendingSearch>();
  private nextWorker = 0;
  private reqSeq = 0;
  private closed = false;

  constructor(opts: SearchPoolOptions) {
    this.store = opts.store;
    const count = opts.workers ?? Math.min(4, Math.max(1, availableParallelism() - 1));
    const n = Math.max(1, Math.trunc(count));
    this.workers = Array.from({ length: n }, () => this.spawn());
  }

  private spawn(): Worker {
    // Self-entry worker: this module runs again in the worker thread and takes
    // the !isMainThread branch above. Initial snapshot travels via workerData.
    const worker = new Worker(pathToFileURL(import.meta.filename), {
      workerData: { data: this.store.index.serialize() },
    });
    worker.on('message', (msg: WorkerOutbound) => {
      if (msg.type === 'result') {
        this.settle(msg.reqId, (p) => p.resolve(msg.results));
      } else {
        const err = new Error(`[skipverse] search worker failed: ${msg.message}`);
        if (msg.reqId !== undefined) this.settle(msg.reqId, (p) => p.reject(err));
        else this.failWorker(worker, err);
      }
    });
    worker.on('error', (err) => this.failWorker(worker, err instanceof Error ? err : new Error(String(err))));
    worker.on('exit', (code) => {
      const i = this.workers.indexOf(worker);
      if (i >= 0) this.workers.splice(i, 1);
      if (!this.closed) this.failWorker(worker, new Error(`[skipverse] search worker exited unexpectedly (code ${code})`));
    });
    return worker;
  }

  /**
   * Search a read-only snapshot in a worker thread. `vec` is copied to a plain
   * number[] so it crosses the thread boundary cleanly. Results come from the
   * last snapshot the receiving worker holds — see {@link refresh}.
   */
  searchAsync(vec: ArrayLike<number>, k = 10, ef?: number): Promise<SearchResult[]> {
    if (this.closed || this.workers.length === 0) {
      return Promise.reject(new Error('[skipverse] SearchPool is closed'));
    }
    const worker = this.workers[this.nextWorker++ % this.workers.length]!;
    const reqId = this.reqSeq++;
    return new Promise<SearchResult[]>((resolve, reject) => {
      this.pending.set(reqId, { resolve, reject, worker });
      worker.postMessage({ type: 'search', reqId, vec: Array.from(vec), k, ef } satisfies WorkerInbound);
    });
  }

  /**
   * Re-serialize the store's current index and broadcast it to every worker.
   * Never called automatically: after upsert/remove/compact you must call
   * refresh() or the read pool keeps serving the previous snapshot.
   */
  refresh(): void {
    if (this.closed) throw new Error('[skipverse] SearchPool is closed');
    const data = this.store.index.serialize();
    for (const worker of this.workers) worker.postMessage({ type: 'load', data } satisfies WorkerInbound);
  }

  /** Ask every worker to drain its queue and exit; resolves when all are gone. */
  async close(): Promise<void> {
    this.closed = true;
    const workers = this.workers.splice(0);
    await Promise.all(
      workers.map(
        (worker) =>
          new Promise<void>((resolve) => {
            worker.once('exit', () => resolve());
            worker.postMessage({ type: 'close' } satisfies WorkerInbound);
          }),
      ),
    );
  }

  private settle(reqId: number, fn: (p: PendingSearch) => void): void {
    const p = this.pending.get(reqId);
    if (!p) return;
    this.pending.delete(reqId);
    fn(p);
  }

  private failWorker(worker: Worker, err: Error): void {
    for (const [reqId, p] of this.pending) {
      if (p.worker === worker) {
        p.reject(err);
        this.pending.delete(reqId);
      }
    }
  }
}

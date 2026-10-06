import { performance } from 'node:perf_hooks';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { VectorStore } from '../src/node/store.js';
import { genClusterData } from '../src/core/dataset.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sv-import-bench-'));
const { vecs } = genClusterData(5000, 64, 20, 7);
const store = VectorStore.open({ dataDir: dir, dim: 64, metric: 'euclidean', M: 12, checkpointEvery: 0 });
const t0 = performance.now();
store.upsertBatch(vecs.map((v, i) => ({ id: String(i), vec: v })));
const ms = performance.now() - t0;
const walSize = fs.statSync(path.join(dir, 'wal.log')).size;
store.close();
console.log(`upsertBatch 5000×64d: ${ms.toFixed(0)}ms, wal ${(walSize / 1048576).toFixed(1)}MB`);
fs.rmSync(dir, { recursive: true, force: true });

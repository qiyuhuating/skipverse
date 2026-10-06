/**
 * skipverse in 60 seconds: build an index, search it, persist it, quantize it.
 *
 *   npx tsx examples/quickstart.ts
 */
import * as fs from 'node:fs';
import { HnswIndex, VectorStore } from '../src/node/index.js';

// tiny toy "embeddings": 3 documents on a 4-dimensional grid
const docs: { id: string; vec: number[]; text: string }[] = [
  { id: 'doc-1', vec: [1, 0, 0, 0], text: 'alpha' },
  { id: 'doc-2', vec: [0, 1, 0, 0], text: 'beta' },
  { id: 'doc-3', vec: [0.9, 0.1, 0, 0], text: 'almost alpha' },
];

// 1) embeddable index --------------------------------------------------------
const idx = new HnswIndex({ dim: 4, metric: 'cosine' });
for (const d of docs) idx.add(d.id, d.vec);
console.log('in-memory search "alpha":', idx.search([1, 0, 0, 0], 2));

// 2) durable store with quantization + tracing search ------------------------
const dir = 'examples/.quickstart-data';
fs.rmSync(dir, { recursive: true, force: true });
const store = VectorStore.open({ dataDir: dir, dim: 4, metric: 'cosine', quantization: 'sq8' });
store.upsertBatch(docs.map((d) => ({ id: d.id, vec: d.vec })));
store.calibrate(); // freeze ranges, rewrite codes, rotate a snapshot
console.log('bytes/vec after sq8 calibration:', store.index.bytesPerVector);

const { results, trace } = store.searchWithTrace([1, 0, 0, 0], 2, { ef: 32 });
console.log('store search:', results);
console.log('the search inspected', trace.visitedTotal, 'nodes across', trace.layers.length, 'layers');

// 3) close, reopen, same answers --------------------------------------------
store.close();
const reopened = VectorStore.open({ dataDir: dir, dim: 4, metric: 'cosine', quantization: 'sq8' });
console.log('reopened search:', reopened.search([1, 0, 0, 0], 2));
console.log(
  'stored vector for doc-1 (dequantized):',
  Array.from(reopened.get('doc-1')!).map((x) => x.toFixed(2)),
);
reopened.close();
fs.rmSync(dir, { recursive: true, force: true });

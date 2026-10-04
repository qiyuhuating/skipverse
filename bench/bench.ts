import { performance } from 'node:perf_hooks';
import { HnswIndex } from '../src/core/hnsw.js';
import type { SearchResult } from '../src/core/types.js';
import { genClusterData } from '../src/core/dataset.js';
import { bruteForce, recall } from '../test/helpers.js';
import type { HnswIndex as Index } from '../src/core/hnsw.js';

interface BenchConfig {
  n: number;
  dim: number;
  clusters: number;
  numQueries: number;
  k: number;
  rounds: number;
  seed: number;
}

interface ModeResult {
  mode: 'f32' | 'sq8';
  buildMs: number;
  bytesPerVector: number;
  results: { ef: number; recall: number; qps: number; visitedAvg: number }[];
}

const DEFAULTS: BenchConfig = { n: 10_000, dim: 64, clusters: 20, numQueries: 100, k: 10, rounds: 5, seed: 1234 };
const SMALL: BenchConfig = { n: 4_000, dim: 32, clusters: 25, numQueries: 50, k: 10, rounds: 3, seed: 1234 };

function buildIndex(cfg: BenchConfig, mode: 'f32' | 'sq8'): Index {
  const { vecs } = genClusterData(cfg.n, cfg.dim, cfg.clusters, cfg.seed);
  const idx = new HnswIndex({
    dim: cfg.dim,
    metric: 'euclidean',
    M: 16,
    efConstruction: 200,
    seed: 42,
    quantization: mode === 'sq8' ? 'sq8' : 'none',
  });
  for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
  if (mode === 'sq8') idx.calibrate();
  return idx;
}

function benchMode(cfg: BenchConfig, mode: 'f32' | 'sq8', truth: SearchResult[][]): ModeResult {
  const { vecs, queries } = genClusterData(cfg.n, cfg.dim, cfg.clusters, cfg.seed);
  const t0 = performance.now();
  const idx = buildIndex(cfg, mode);
  const buildMs = performance.now() - t0;

  const qs = queries.slice(0, cfg.numQueries);
  const results: ModeResult['results'] = [];
  for (const ef of [16, 64, 128]) {
    const t1 = performance.now();
    const answers: SearchResult[][] = [];
    for (let r = 0; r < cfg.rounds; r++) {
      for (const q of qs) answers.push(idx.search(q, cfg.k, { ef }));
    }
    const dt = (performance.now() - t1) / 1000;
    const rec = answers.slice(0, cfg.numQueries).map((a, i) => recall(truth[i]!, a)).reduce((s, x) => s + x, 0) / cfg.numQueries;
    let visited = 0;
    for (const q of qs) visited += idx.searchWithTrace(q, cfg.k, { ef }).trace.visitedTotal;
    results.push({ ef, recall: rec, qps: (cfg.numQueries * cfg.rounds) / dt, visitedAvg: visited / cfg.numQueries });
  }
  return { mode, buildMs, bytesPerVector: idx.bytesPerVector, results };
}

function main(): void {
  const argv = process.argv.slice(2);
  const small = argv.includes('--small');
  const assertMode = argv.includes('--assert');
  const cfg = small ? SMALL : DEFAULTS;

  const { vecs, queries } = genClusterData(cfg.n, cfg.dim, cfg.clusters, cfg.seed);
  const qs = queries.slice(0, cfg.numQueries);
  const truth = qs.map((q) => bruteForce(vecs, q, cfg.k, 'euclidean'));

  const f32 = benchMode(cfg, 'f32', truth);
  const sq8 = benchMode(cfg, 'sq8', truth);

  const t2 = performance.now();
  for (let r = 0; r < cfg.rounds; r++) {
    for (const q of qs) bruteForce(vecs, q, cfg.k, 'euclidean');
  }
  const bruteQps = (cfg.numQueries * cfg.rounds) / ((performance.now() - t2) / 1000);

  const lines = [
    `### skipverse ${cfg.n.toLocaleString()} × ${cfg.dim}d · M=16 · efConstruction=200 · euclidean · k=${cfg.k}`,
    '',
    '| mode | efSearch | recall@10 | QPS | avg nodes visited |',
    '|:-----|---------:|----------:|----:|------------------:|',
    ...f32.results.map((x) => `| f32 | ${x.ef} | ${x.recall.toFixed(4)} | ${Math.round(x.qps).toLocaleString()} | ${x.visitedAvg.toFixed(1)} |`),
    ...sq8.results.map((x) => `| sq8 | ${x.ef} | ${x.recall.toFixed(4)} | ${Math.round(x.qps).toLocaleString()} | ${x.visitedAvg.toFixed(1)} |`),
    `| brute force | — | 1.0000 | ${Math.round(bruteQps).toLocaleString()} | ${cfg.n.toLocaleString()} |`,
    '',
    `build: f32 ${(f32.buildMs / 1000).toFixed(2)}s · sq8 ${(sq8.buildMs / 1000).toFixed(2)}s (incl. calibration)`,
    `vector storage: f32 ${f32.bytesPerVector} B/vec · sq8 ${sq8.bytesPerVector} B/vec · ${(f32.bytesPerVector / sq8.bytesPerVector).toFixed(1)}× smaller`,
  ];
  console.log(lines.join('\n'));

  if (assertMode) {
    const f64 = f32.results.find((x) => x.ef === 64)!;
    const s64 = sq8.results.find((x) => x.ef === 64)!;
    let ok = true;
    if (f64.recall < 0.95) {
      console.error(`FAIL: f32 recall@10 @ef64 = ${f64.recall.toFixed(4)} < 0.95`);
      ok = false;
    } else {
      console.log(`OK: f32 recall@10 @ef64 = ${f64.recall.toFixed(4)} ≥ 0.95`);
    }
    if (s64.recall < 0.9) {
      console.error(`FAIL: sq8 recall@10 @ef64 = ${s64.recall.toFixed(4)} < 0.90`);
      ok = false;
    } else {
      console.log(`OK: sq8 recall@10 @ef64 = ${s64.recall.toFixed(4)} ≥ 0.90`);
    }
    if (!ok) process.exit(1);
  }
}

main();

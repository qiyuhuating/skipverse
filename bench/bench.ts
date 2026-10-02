import { performance } from 'node:perf_hooks';
import { HnswIndex } from '../src/core/hnsw.js';
import { genClusterData } from '../src/core/dataset.js';
import { bruteForce, recall } from '../test/helpers.js';

interface BenchConfig {
  n: number;
  dim: number;
  clusters: number;
  numQueries: number;
  k: number;
  rounds: number;
  seed: number;
}

interface EfResult {
  ef: number;
  recall: number;
  qps: number;
  visitedAvg: number;
}

const DEFAULTS: BenchConfig = { n: 10_000, dim: 64, clusters: 20, numQueries: 100, k: 10, rounds: 5, seed: 1234 };
const SMALL: BenchConfig = { n: 4_000, dim: 32, clusters: 25, numQueries: 50, k: 10, rounds: 3, seed: 1234 };

function bench(cfg: BenchConfig): { buildMs: number; results: EfResult[]; bruteQps: number } {
  const { vecs, queries } = genClusterData(cfg.n, cfg.dim, cfg.clusters, cfg.seed);
  const idx = new HnswIndex({ dim: cfg.dim, metric: 'euclidean', M: 16, efConstruction: 200, seed: 42 });

  const t0 = performance.now();
  for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
  const buildMs = performance.now() - t0;

  const truth = queries.slice(0, cfg.numQueries).map((q) => bruteForce(vecs, q, cfg.k, 'euclidean'));

  const results: EfResult[] = [];
  for (const ef of [16, 64, 128]) {
    const t1 = performance.now();
    const answers: { id: string; dist: number }[][] = [];
    for (let r = 0; r < cfg.rounds; r++) {
      for (const q of queries.slice(0, cfg.numQueries)) answers.push(idx.search(q, cfg.k, { ef }));
    }
    const dt = (performance.now() - t1) / 1000;
    const rec = answers
      .slice(0, cfg.numQueries)
      .map((a, i) => recall(truth[i]!, a))
      .reduce((s, x) => s + x, 0) / cfg.numQueries;
    let visited = 0;
    for (const q of queries.slice(0, cfg.numQueries)) visited += idx.searchWithTrace(q, cfg.k, { ef }).trace.visitedTotal;
    results.push({ ef, recall: rec, qps: (cfg.numQueries * cfg.rounds) / dt, visitedAvg: visited / cfg.numQueries });
  }

  const t2 = performance.now();
  for (let r = 0; r < cfg.rounds; r++) {
    for (const q of queries.slice(0, cfg.numQueries)) bruteForce(vecs, q, cfg.k, 'euclidean');
  }
  const bruteQps = (cfg.numQueries * cfg.rounds) / ((performance.now() - t2) / 1000);

  return { buildMs, results, bruteQps };
}

function table(cfg: BenchConfig, r: ReturnType<typeof bench>): string {
  const lines = [
    `### skipverse ${cfg.n.toLocaleString()} × ${cfg.dim}d · M=16 · efConstruction=200 · euclidean · k=${cfg.k}`,
    '',
    '| efSearch | recall@10 | QPS | avg nodes visited |',
    '|---------:|----------:|----:|------------------:|',
    ...r.results.map(
      (x) => `| ${x.ef} | ${x.recall.toFixed(4)} | ${Math.round(x.qps).toLocaleString()} | ${x.visitedAvg.toFixed(1)} |`,
    ),
    `| brute force | 1.0000 | ${Math.round(r.bruteQps).toLocaleString()} | ${cfg.n.toLocaleString()} |`,
    '',
    `build: ${(r.buildMs / 1000).toFixed(2)}s for ${cfg.n.toLocaleString()} vectors (seed ${cfg.seed}, deterministic)`,
  ];
  return lines.join('\n');
}

const argv = process.argv.slice(2);
const small = argv.includes('--small');
const assertMode = argv.includes('--assert');
const cfg = small ? SMALL : DEFAULTS;
const result = bench(cfg);
console.log(table(cfg, result));

if (assertMode) {
  const r64 = result.results.find((x) => x.ef === 64)!;
  if (r64.recall < 0.95) {
    console.error(`\nFAIL: recall@10 @ef64 = ${r64.recall.toFixed(4)} < 0.95`);
    process.exit(1);
  }
  console.log(`\nOK: recall@10 @ef64 = ${r64.recall.toFixed(4)} ≥ 0.95`);
}

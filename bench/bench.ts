import { performance } from 'node:perf_hooks';
import { HnswIndex } from '../src/core/hnsw.js';
import type { SearchResult } from '../src/core/types.js';
import type { Quantization } from '../src/core/hnsw.js';
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

type Mode = 'f32' | 'sq8' | 'sq4';

interface ModeResult {
  mode: Mode;
  buildMs: number;
  bytesPerVector: number;
  results: { ef: number; recall: number; qps: number; visitedAvg: number }[];
}

const DEFAULTS: BenchConfig = { n: 10_000, dim: 64, clusters: 20, numQueries: 100, k: 10, rounds: 5, seed: 1234 };
const SMALL: BenchConfig = { n: 4_000, dim: 32, clusters: 25, numQueries: 50, k: 10, rounds: 3, seed: 1234 };
const MODES: Mode[] = ['f32', 'sq8', 'sq4'];
const quantizationOf = (m: Mode): Quantization | undefined => (m === 'f32' ? undefined : m);

function buildIndex(cfg: BenchConfig, mode: Mode): Index {
  const { vecs } = genClusterData(cfg.n, cfg.dim, cfg.clusters, cfg.seed);
  const idx = new HnswIndex({
    dim: cfg.dim,
    metric: 'euclidean',
    M: 16,
    efConstruction: 200,
    seed: 42,
    quantization: quantizationOf(mode),
  });
  for (let i = 0; i < vecs.length; i++) idx.add(String(i), vecs[i]!);
  if (mode !== 'f32') idx.calibrate();
  return idx;
}

function benchMode(cfg: BenchConfig, mode: Mode, truth: SearchResult[][]): ModeResult {
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

  const runs = MODES.map((m) => benchMode(cfg, m, truth));

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
    ...runs.flatMap((run) =>
      run.results.map((x) => `| ${run.mode} | ${x.ef} | ${x.recall.toFixed(4)} | ${Math.round(x.qps).toLocaleString()} | ${x.visitedAvg.toFixed(1)} |`),
    ),
    `| brute force | — | 1.0000 | ${Math.round(bruteQps).toLocaleString()} | ${cfg.n.toLocaleString()} |`,
    '',
    `build: ${runs.map((r) => `${r.mode} ${(r.buildMs / 1000).toFixed(2)}s`).join(' · ')}`,
    `vector storage: ${runs.map((r) => `${r.mode} ${r.bytesPerVector} B/vec`).join(' · ')} · f32/sq8 = ${(runs[0]!.bytesPerVector / runs[1]!.bytesPerVector).toFixed(1)}× · f32/sq4 = ${(runs[0]!.bytesPerVector / runs[2]!.bytesPerVector).toFixed(1)}×`,
  ];
  console.log(lines.join('\n'));

  if (assertMode) {
    let ok = true;
    const gates: [Mode, number, number][] = [
      ['f32', 0.95, 64],
      ['sq8', 0.9, 64],
      ['sq4', 0.5, 64],
    ];
    for (const [mode, floor, ef] of gates) {
      const run = runs.find((r) => r.mode === mode)!;
      const at = run.results.find((x) => x.ef === ef)!;
      if (at.recall < floor) {
        console.error(`FAIL: ${mode} recall@10 @ef${ef} = ${at.recall.toFixed(4)} < ${floor}`);
        ok = false;
      } else {
        console.log(`OK: ${mode} recall@10 @ef${ef} = ${at.recall.toFixed(4)} ≥ ${floor}`);
      }
    }
    if (!ok) process.exit(1);
  }
}

main();

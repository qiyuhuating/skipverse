import { gaussian, mulberry32 } from './rng.js';

export interface Dataset {
  vecs: Float32Array[];
  labels: Uint8Array;
  queries: Float32Array[];
}

/**
 * Deterministic clustered dataset: `clusters` gaussian blobs of stddev 0.7
 * around centers ~N(0, 3), plus `queries` points sampled near the overall
 * distribution. Same seed ⇒ same data, on Node and in the browser.
 */
export function genClusterData(n: number, dim: number, clusters: number, seed: number): Dataset {
  const rng = mulberry32(seed);
  const g = gaussian(rng);
  const centers: Float32Array[] = [];
  for (let c = 0; c < clusters; c++) {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = g() * 3;
    centers.push(v);
  }
  const vecs: Float32Array[] = [];
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % clusters;
    labels[i] = c;
    const v = new Float32Array(dim);
    for (let j = 0; j < dim; j++) v[j] = centers[c]![j]! + g() * 0.7;
    vecs.push(v);
  }
  // queries: in-distribution — a random data point plus small noise, which is
  // how embedding retrieval actually looks (queries land near indexed points)
  const queries: Float32Array[] = [];
  for (let i = 0; i < 100; i++) {
    const src = vecs[Math.floor(rng() * n) % n]!;
    const v = new Float32Array(dim);
    for (let j = 0; j < dim; j++) v[j] = src[j]! + g() * 0.3;
    queries.push(v);
  }
  return { vecs, labels, queries };
}

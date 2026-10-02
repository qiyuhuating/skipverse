export { genClusterData, type Dataset } from '../src/core/dataset.js';
import { makeDistance, type DistanceFn } from '../src/core/distance.js';
import type { Metric } from '../src/core/types.js';

export function bruteForce(vecs: Float32Array[], q: Float32Array, k: number, metric: Metric): { id: string; dist: number }[] {
  const dist = makeDistance(metric);
  const scored = vecs.map((v, i) => ({ id: String(i), dist: dist(v, q) }));
  scored.sort((a, b) => a.dist - b.dist);
  return scored.slice(0, k);
}

export function recall(expected: { id: string }[], got: { id: string }[]): number {
  const gotSet = new Set(got.map((r) => r.id));
  let hit = 0;
  for (const e of expected) {
    if (gotSet.has(e.id)) hit++;
  }
  return expected.length === 0 ? 1 : hit / expected.length;
}

export const distOf = (metric: Metric): DistanceFn => makeDistance(metric);

import type { Metric } from './types.js';

export type DistanceFn = (a: Float32Array, b: Float32Array) => number;

export function makeDistance(metric: Metric): DistanceFn {
  switch (metric) {
    case 'cosine':
      // cosine vectors are L2-normalized at insert time, so 1 − dot is exact
      return (a, b) => 1 - dot(a, b);
    case 'dot':
      // MIPS: smaller distance = larger inner product
      return (a, b) => -dot(a, b);
    case 'euclidean':
      return (a, b) => Math.sqrt(sqEuclidean(a, b));
  }
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

export function sqEuclidean(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    s += d * d;
  }
  return s;
}

export function normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i]! * v[i]!;
  n = Math.sqrt(n);
  const out = new Float32Array(v.length);
  if (n === 0) {
    out.set(v);
    return out;
  }
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / n;
  return out;
}

/** Prepare a query the same way indexed vectors are prepared. */
export function prepareVector(v: ArrayLike<number>, metric: Metric, dim: number): Float32Array {
  if (v.length !== dim) {
    throw new Error(`vector dimension mismatch: expected ${dim}, got ${v.length}`);
  }
  const f = Float32Array.from(v);
  return metric === 'cosine' ? normalize(f) : f;
}

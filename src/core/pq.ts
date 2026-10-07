import { mulberry32 } from './rng.js';

/**
 * Product quantization model: the dim dimensions are split into m segments
 * (each ~dim/m wide); every segment has its own k=256 codebook trained by
 * deterministic k-means. A vector is stored as m one-byte codebook indices —
 * for 64d with m=8 that is 8 bytes per vector, 32× smaller than f32.
 */
export interface PqModel {
  m: number;
  k: number;
  /** dims per subspace (last one may be shorter) */
  segLens: number[];
  /** starting dim of each subspace */
  offsets: number[];
  /** codebooks[i] = Float32Array(k * segLens[i]) */
  codebooks: Float32Array[];
}

/** split dim into m segments: segLen = ceil(dim/m), last may be shorter */
export function pqSegments(dim: number, m: number): { segLens: number[]; offsets: number[] } {
  const seg = Math.ceil(dim / m);
  const segLens: number[] = [];
  const offsets: number[] = [];
  let off = 0;
  while (off < dim) {
    const len = Math.min(seg, dim - off);
    segLens.push(len);
    offsets.push(off);
    off += len;
  }
  return { segLens, offsets };
}

/** squared euclidean distance between a segment of v and a centroid */
function segDist2(v: Float32Array, off: number, len: number, cb: Float32Array, cBase: number): number {
  let s = 0;
  for (let d = 0; d < len; d++) {
    const diff = v[off + d]! - cb[cBase + d]!;
    s += diff * diff;
  }
  return s;
}

/**
 * Deterministic k-means per subspace: k-means++ seeding on mulberry32(seed ^
  * subspace), 12 Lloyd iterations, empty clusters absorb the farthest point.
 */
function kmeansSegment(mat: Float32Array[], off: number, len: number, k: number, seed: number): Float32Array {
  const n = mat.length;
  const rng = mulberry32(seed);
  const centroids = new Float32Array(k * len);

  // k-means++ seeding
  const first = Math.floor(rng() * n) % n;
  centroids.set(mat[first]!.subarray(off, off + len), 0);
  const dist2 = new Float64Array(n).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let sum = 0;
    const prev = (c - 1) * len;
    for (let i = 0; i < n; i++) {
      const d = segDist2(mat[i]!, off, len, centroids, prev);
      if (d < dist2[i]!) dist2[i] = d;
      sum += dist2[i]!;
    }
    // sample next seed ∝ dist²
    let target = rng() * sum;
    let pick = n - 1;
    for (let i = 0; i < n; i++) {
      target -= dist2[i]!;
      if (target <= 0) {
        pick = i;
        break;
      }
    }
    centroids.set(mat[pick]!.subarray(off, off + len), c * len);
  }

  // Lloyd iterations
  const sums = new Float64Array(k * len);
  const counts = new Uint32Array(k);
  for (let iter = 0; iter < 12; iter++) {
    sums.fill(0);
    counts.fill(0);
    let moved = 0;
    for (let i = 0; i < n; i++) {
      const v = mat[i]!;
      let best = 0;
      let bd = Infinity;
      for (let c = 0; c < k; c++) {
        const d = segDist2(v, off, len, centroids, c * len);
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      counts[best]++;
      for (let d = 0; d < len; d++) sums[best * len + d] += v[off + d]!;
      moved += bd;
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        // empty cluster: absorb the point farthest from its assigned centroid
        let far = 0;
        let fd = -1;
        for (let i = 0; i < n; i++) {
          let bd = Infinity;
          for (let c2 = 0; c2 < k; c2++) {
            const d = segDist2(mat[i]!, off, len, centroids, c2 * len);
            if (d < bd) bd = d;
          }
          if (bd > fd) {
            fd = bd;
            far = i;
          }
        }
        centroids.set(mat[far]!.subarray(off, off + len), c * len);
        continue;
      }
      for (let d = 0; d < len; d++) centroids[c * len + d] = sums[c * len + d] / counts[c];
    }
    void moved;
  }
  return centroids;
}

/** Train an m-subspace PQ model with k centroids per subspace. Deterministic. */
export function trainPq(vecs: Float32Array[], dim: number, m: number, k: number, seed: number): PqModel {
  const effM = Math.min(m, dim);
  const effK = Math.max(1, Math.min(k, vecs.length));
  const { segLens, offsets } = pqSegments(dim, effM);
  const codebooks: Float32Array[] = [];
  for (let i = 0; i < effM; i++) {
    codebooks.push(kmeansSegment(vecs, offsets[i]!, segLens[i]!, effK, (seed ^ (0x5051 + i * 0x9e37)) >>> 0));
  }
  return { m: effM, k: effK, segLens, offsets, codebooks };
}

/** Encode one vector: m bytes, each the nearest-centroid index of its subspace. */
export function encodePq(model: PqModel, v: Float32Array): Uint8Array {
  const codes = new Uint8Array(model.m);
  for (let i = 0; i < model.m; i++) {
    const len = model.segLens[i]!;
    const off = model.offsets[i]!;
    const cb = model.codebooks[i]!;
    let best = 0;
    let bd = Infinity;
    for (let c = 0; c < model.k; c++) {
      const d = segDist2(v, off, len, cb, c * len);
      if (d < bd) {
        bd = d;
        best = c;
      }
    }
    codes[i] = best;
  }
  return codes;
}

/** Rebuild an approximate f32 vector from codes. */
export function reconstructPq(model: PqModel, codes: Uint8Array): Float32Array {
  const out = new Float32Array(model.offsets[model.m - 1]! + model.segLens[model.m - 1]!);
  for (let i = 0; i < model.m; i++) {
    const c = codes[i]!;
    out.set(
      model.codebooks[i]!.subarray(c * model.segLens[i]!, (c + 1) * model.segLens[i]!),
      model.offsets[i]!,
    );
  }
  return out;
}

/**
 * Symmetric PQ distance between two coded vectors (node-node path):
 * reconstruct a in-register, then per-subspace distance/dot against b's centroids.
 */
export function distPqSym(
  model: PqModel,
  aCodes: Uint8Array,
  bCodes: Uint8Array,
  metric: 'euclidean' | 'cosine' | 'dot',
  normA: number,
  normB: number,
): number {
  let s = 0;
  for (let i = 0; i < model.m; i++) {
    const len = model.segLens[i]!;
    const off = model.offsets[i]!;
    const cb = model.codebooks[i]!;
    const ca = aCodes[i]! * len;
    const cbB = bCodes[i]!;
    if (metric === 'euclidean') {
      for (let d = 0; d < len; d++) {
        const diff = cb[ca + d]! - cb[cbB * len + d]!;
        s += diff * diff;
      }
    } else {
      for (let d = 0; d < len; d++) s += cb[ca + d]! * cb[cbB * len + d]!;
    }
  }
  if (metric === 'euclidean') return Math.sqrt(s);
  if (metric === 'dot') return -s;
  const denom = normA * normB;
  return denom === 0 ? 1 : 1 - s / denom;
}

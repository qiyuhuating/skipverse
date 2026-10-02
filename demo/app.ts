import { HnswIndex } from '../src/core/hnsw.js';
import type { SearchTrace } from '../src/core/types.js';

// --------------------------------------------------------------------- state

const PALETTE = ['#f472b6', '#60a5fa', '#34d399', '#fbbf24', '#a78bfa', '#fb7185', '#22d3ee', '#facc15'];

interface Hop { layer: number; from: number; to: number; tStart: number; tEnd: number }
interface Point { x: number; y: number; cluster: number }

let points: Point[] = [];
let idx: HnswIndex | null = null;
let buildMs = 0;
let trace: SearchTrace | null = null;
let hops: Hop[] = [];
let animStart = 0;
let animating = false;
let query: { x: number; y: number } | null = null;
let resultIds: string[] = [];
let serverMode = false;
let layerVisible: boolean[] = [];
let cachedEdges: number[][][] | null = null; // per level: list of [a, b] index pairs

const $ = (id: string): HTMLElement => document.getElementById(id)!;
const canvas = $('cv') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const DATA_RANGE = 11; // data lives in roughly [-10, 10]
const CLUSTERS = 5;
const PER_CLUSTER = 160;

function rngFrom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ geometry

function dataToScreen(x: number, y: number): [number, number] {
  const w = canvas.width / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  const s = ((Math.min(w, h) / (2 * DATA_RANGE)) * 0.94);
  return [w / 2 + x * s, h / 2 - y * s];
}

function screenToData(sx: number, sy: number): [number, number] {
  const w = canvas.width / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  const s = ((Math.min(w, h) / (2 * DATA_RANGE)) * 0.94);
  return [(sx - w / 2) / s, (h / 2 - sy) / s];
}

function resize(): void {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * devicePixelRatio);
  canvas.height = Math.round(r.height * devicePixelRatio);
}
window.addEventListener('resize', resize);

// --------------------------------------------------------------------- build

function rebuildIndex(): void {
  const M = Number(($('s-m') as HTMLInputElement).value);
  const efC = Number(($('s-efc') as HTMLInputElement).value);
  idx = new HnswIndex({ dim: 2, metric: 'euclidean', M, efConstruction: efC, seed: 7 });
  const t0 = performance.now();
  for (let i = 0; i < points.length; i++) idx.add(String(i), [points[i]!.x, points[i]!.y]);
  buildMs = performance.now() - t0;
  trace = null;
  resultIds = [];
  cachedEdges = null;
  buildLayerToggles();
  renderIndexStats();
}

function regenerate(): void {
  const rng = rngFrom(0x5ea);
  const centers: [number, number][] = [];
  for (let c = 0; c < CLUSTERS; c++) {
    const ang = (c / CLUSTERS) * Math.PI * 2 + rng() * 0.5;
    const rad = 4 + rng() * 4.5;
    centers.push([Math.cos(ang) * rad, Math.sin(ang) * rad]);
  }
  points = [];
  for (let i = 0; i < CLUSTERS * PER_CLUSTER; i++) {
    const [cx, cy] = centers[i % CLUSTERS]!;
    points.push({
      x: cx + (rng() * 2 - 1) * 2.4,
      y: cy + (rng() * 2 - 1) * 2.4,
      cluster: i % CLUSTERS,
    });
  }
  rebuildIndex();
}

function buildLayerToggles(): void {
  const stats = idx!.stats();
  layerVisible = stats.levels.map(() => true);
  const list = $('layer-list');
  list.innerHTML = '';
  [...stats.levels].reverse().forEach((l) => {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.addEventListener('change', () => {
      layerVisible[l.level] = cb.checked;
    });
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = l.level === 0 ? '#94a3b8' : '#38bdf8';
    const name = document.createTextNode(` layer ${l.level}`);
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = `${l.nodes} nodes · ${l.avgDegree.toFixed(1)}°`;
    label.append(cb, dot, name, count);
    list.append(label);
  });
}

function renderIndexStats(): void {
  const stats = idx!.stats();
  $('index-stats').innerHTML = [
    ['vectors', String(stats.count)],
    ['build', `${buildMs.toFixed(0)} ms`],
    ['max layer', String(stats.maxLevel)],
    ['params', `M=${stats.params.M}`],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
}

function computeEdges(): number[][][] {
  const maxLevel = idx!.stats().maxLevel;
  const edges: number[][][] = Array.from({ length: maxLevel + 1 }, () => []);
  for (let i = 0; i < points.length; i++) {
    const adj = idx!.adjacency(String(i));
    if (adj === null) continue;
    for (let l = 0; l < adj.length; l++) {
      for (const nb of adj[l]!) {
        const j = Number(nb);
        if (i < j) edges[l]!.push([i, j]);
      }
    }
  }
  return edges;
}

// -------------------------------------------------------------------- search

async function runSearch(q: { x: number; y: number }): Promise<void> {
  if (idx === null) return;
  query = q;
  const k = Number(($('s-k') as HTMLInputElement).value);
  const ef = Number(($('s-ef') as HTMLInputElement).value);
  const t0 = performance.now();
  let results: { id: string; dist: number }[];
  let t: SearchTrace;
  if (serverMode) {
    const res = await fetch('/search', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ vec: [q.x, q.y], k, ef, trace: true }),
    });
    const body = (await res.json()) as { results: typeof results; trace: SearchTrace };
    results = body.results;
    t = body.trace;
  } else {
    const r = idx.searchWithTrace([q.x, q.y], k, { ef });
    results = r.results;
    t = r.trace;
  }
  const elapsed = performance.now() - t0;
  trace = t;
  resultIds = results.map((r) => r.id);
  // flatten hops into an animation timeline: upper layers first, then layer 0
  hops = [];
  let acc = 0;
  const per = Number(($('s-speed') as HTMLInputElement).value);
  for (const layer of t.layers) {
    for (const hop of layer.hops) {
      hops.push({ layer: layer.level, from: Number(hop.from), to: Number(hop.to), tStart: acc, tEnd: acc + per });
      acc += per;
    }
    acc += per * 0.6; // beat between layers
  }
  animStart = performance.now();
  animating = hops.length > 0;
  renderTraceStats(elapsed, results, t);
}

function renderTraceStats(elapsed: number, results: { id: string; dist: number }[], t: SearchTrace): void {
  const totalHops = t.layers.reduce((s, l) => s + l.hops.length, 0);
  $('trace-stats').innerHTML = [
    ['layers used', String(t.layers.length)],
    ['edges inspected', String(totalHops)],
    ['nodes visited', String(t.visitedTotal)],
    ['latency', `${elapsed.toFixed(2)} ms`],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
  $('results').innerHTML = results
    .map((r, i) => {
      const cluster = points[Number(r.id)]!.cluster;
      return `<div class="r"><span class="rk">${i + 1}</span><span class="dot" style="background:${PALETTE[cluster % PALETTE.length]}"></span><span>#${r.id}</span><span class="rd">${r.dist.toFixed(3)}</span></div>`;
    })
    .join('');
}

// -------------------------------------------------------------------- render

function easeInOut(u: number): number {
  return u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
}

function currentVisited(): Set<number> {
  const out = new Set<number>();
  if (trace === null) return out;
  if (!animating) {
    for (const layer of trace.layers) {
      for (const hp of layer.hops) out.add(Number(hp.to));
    }
    return out;
  }
  const t = performance.now() - animStart;
  for (const hp of hops) {
    if (hp.tEnd <= t) out.add(hp.to);
  }
  return out;
}

function draw(now: number): void {
  const w = canvas.width / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.fillStyle = '#07090f';
  ctx.fillRect(0, 0, w, h);

  // faint grid
  ctx.strokeStyle = 'rgba(148,163,184,0.05)';
  ctx.lineWidth = 1;
  for (let g = -10; g <= 10; g += 5) {
    const [x0, y0] = dataToScreen(g, -10);
    const [x1, y1] = dataToScreen(g, 10);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const [x2, y2] = dataToScreen(-10, g);
    const [x3, y3] = dataToScreen(10, g);
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
  }

  if (idx !== null) {
    const maxLevel = idx.stats().maxLevel;
    if (cachedEdges === null) cachedEdges = computeEdges();

    // edges per visible layer (higher layers brighter blue, layer 0 grey)
    for (let l = maxLevel; l >= 0; l--) {
      if (layerVisible[l] === false) continue;
      ctx.strokeStyle = l === 0 ? 'rgba(148,163,184,0.10)' : `rgba(56,189,248,${0.2 + 0.12 * l})`;
      ctx.lineWidth = l === 0 ? 1 : 1.4;
      ctx.beginPath();
      for (const [a, b] of cachedEdges[l]!) {
        const pa = points[a]!;
        const pb = points[b]!;
        const [ax, ay] = dataToScreen(pa.x, pa.y);
        const [bx, by] = dataToScreen(pb.x, pb.y);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }
  }

  // points + visited glow
  const visitedNow = currentVisited();
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    const [sx, sy] = dataToScreen(p.x, p.y);
    const visited = visitedNow.has(i);
    ctx.fillStyle = PALETTE[p.cluster % PALETTE.length];
    ctx.globalAlpha = visited ? 1 : 0.72;
    ctx.beginPath();
    ctx.arc(sx, sy, visited ? 3.4 : 2.4, 0, Math.PI * 2);
    ctx.fill();
    if (visited) {
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      ctx.arc(sx, sy, 7, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  // animated search: recent traversed edges + travelling token
  if (trace !== null && animating) {
    const t = now - animStart;
    const done = hops.filter((hp) => hp.tEnd <= t);
    for (const hp of done.slice(-12)) {
      const a = points[hp.from]!;
      const b = points[hp.to]!;
      const [ax, ay] = dataToScreen(a.x, a.y);
      const [bx, by] = dataToScreen(b.x, b.y);
      ctx.strokeStyle = 'rgba(52,211,153,0.5)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    const active = hops.find((hp) => t >= hp.tStart && t < hp.tEnd);
    if (active !== undefined) {
      const a = points[active.from]!;
      const b = points[active.to]!;
      const u = easeInOut((t - active.tStart) / (active.tEnd - active.tStart));
      const [ax, ay] = dataToScreen(a.x, a.y);
      const [bx, by] = dataToScreen(b.x, b.y);
      const x = ax + (bx - ax) * u;
      const y = ay + (by - ay) * u;
      ctx.strokeStyle = 'rgba(52,211,153,0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = '#34d399';
      ctx.shadowColor = '#34d399';
      ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
    }
    if (t > (hops[hops.length - 1]?.tEnd ?? 0) + 200) animating = false;
  }

  // final results
  if (trace !== null && !animating && resultIds.length > 0) {
    for (const id of resultIds) {
      const p = points[Number(id)]!;
      const [sx, sy] = dataToScreen(p.x, p.y);
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx, sy, 8, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // query marker
  if (query !== null) {
    const [sx, sy] = dataToScreen(query.x, query.y);
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx - 12, sy); ctx.lineTo(sx - 5, sy);
    ctx.moveTo(sx + 5, sy); ctx.lineTo(sx + 12, sy);
    ctx.moveTo(sx, sy - 12); ctx.lineTo(sx, sy - 5);
    ctx.moveTo(sx, sy + 5); ctx.lineTo(sx, sy + 12);
    ctx.stroke();
  }
}

// --------------------------------------------------------------------- wiring

canvas.addEventListener('click', (ev) => {
  const r = canvas.getBoundingClientRect();
  const [x, y] = screenToData(ev.clientX - r.left, ev.clientY - r.top);
  void runSearch({ x, y });
});
$('b-random').addEventListener('click', () => {
  const p = points[Math.floor(Math.random() * points.length)]!;
  void runSearch({ x: p.x + (Math.random() - 0.5) * 1.2, y: p.y + (Math.random() - 0.5) * 1.2 });
});
$('b-rebuild').addEventListener('click', rebuildIndex);
for (const [slider, out, suffix] of [
  ['s-m', 'v-m', ''],
  ['s-efc', 'v-efc', ''],
  ['s-k', 'v-k', ''],
  ['s-ef', 'v-ef', ''],
  ['s-speed', 'v-speed', 'ms'],
] as const) {
  $(slider).addEventListener('input', () => {
    $(out).textContent = ($(slider) as HTMLInputElement).value + suffix;
  });
}

async function detectEngine(): Promise<void> {
  const badge = $('engine-badge');
  try {
    const r = await fetch('/healthz', { signal: AbortSignal.timeout(1200) });
    if (r.ok) {
      serverMode = true;
      badge.textContent = 'ENGINE: SERVER';
      badge.classList.add('server');
      // an empty server index gets the demo dataset pushed into it —
      // which doubles as a live demonstration of the HTTP API
      const { count } = (await r.json()) as { count: number };
      if (count === 0) {
        await fetch('/vectors', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            vectors: points.map((p, i) => ({ id: String(i), vec: [p.x, p.y] })),
          }),
        });
      }
      return;
    }
  } catch {
    /* not served by skipverse — run fully in-browser */
  }
  badge.textContent = 'ENGINE: BROWSER';
}

resize();
regenerate();
void detectEngine();
requestAnimationFrame(loop);
function loop(): void {
  draw(performance.now());
  requestAnimationFrame(loop);
}

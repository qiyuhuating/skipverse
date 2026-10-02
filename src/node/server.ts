import * as http from 'node:http';
import * as fs from 'node:fs';
import * as path from 'node:path';
import type { VectorStore } from './store.js';

const MAX_BODY = 48 * 1024 * 1024;

export interface ServerOptions {
  store: VectorStore;
  /** directory containing demo/index.html + bundle.js; enables GET / */
  demoDir?: string;
}

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

function send(res: http.ServerResponse, code: number, body: string | Uint8Array, type = 'application/json; charset=utf-8'): void {
  res.writeHead(code, { 'content-type': type, 'content-length': body.length });
  res.end(body);
}

function json(res: http.ServerResponse, code: number, obj: unknown): void {
  send(res, code, JSON.stringify(obj));
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(new Error('body too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

export function createServer(opts: ServerOptions): http.Server {
  const { store, demoDir } = opts;

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    const route = `${req.method} ${url.pathname}`;
    try {
      if (req.method === 'GET' && url.pathname === '/healthz') {
        return json(res, 200, { ok: true, count: store.index.size });
      }
      if (req.method === 'GET' && url.pathname === '/stats') {
        return json(res, 200, store.info());
      }
      if (req.method === 'POST' && url.pathname === '/vectors') {
        const body = JSON.parse(await readBody(req)) as { vectors?: { id: string; vec: number[] }[] };
        if (!Array.isArray(body.vectors)) return json(res, 400, { error: 'expected { vectors: [{id, vec}] }' });
        store.upsertBatch(body.vectors);
        return json(res, 200, { upserted: body.vectors.length, count: store.index.size });
      }
      if (req.method === 'POST' && url.pathname === '/search') {
        const body = JSON.parse(await readBody(req)) as {
          vec?: number[];
          k?: number;
          ef?: number;
          trace?: boolean;
        };
        if (!Array.isArray(body.vec)) return json(res, 400, { error: 'expected { vec: number[], k?, ef?, trace? }' });
        const k = Math.min(Math.max(body.k ?? 10, 1), 1000);
        if (body.trace) {
          const { results, trace } = store.searchWithTrace(body.vec, k, { ef: body.ef });
          return json(res, 200, { results, trace });
        }
        return json(res, 200, { results: store.search(body.vec, k, { ef: body.ef }) });
      }
      if (req.method === 'DELETE' && url.pathname.startsWith('/vectors/')) {
        const id = decodeURIComponent(url.pathname.slice('/vectors/'.length));
        return json(res, 200, { removed: store.remove(id) });
      }
      if (req.method === 'POST' && url.pathname === '/checkpoint') {
        store.checkpoint();
        return json(res, 200, { ok: true });
      }
      if (req.method === 'GET' && demoDir !== undefined) {
        const rel = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
        const file = path.normalize(path.join(demoDir, rel));
        if (file.startsWith(path.normalize(demoDir)) && fs.existsSync(file) && fs.statSync(file).isFile()) {
          const type = MIME[path.extname(file)] ?? 'application/octet-stream';
          return send(res, 200, fs.readFileSync(file), type);
        }
      }
      return json(res, 404, { error: `no route: ${route}` });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const code = msg.includes('dimension mismatch') || msg.includes('expected') ? 400 : 500;
      return json(res, code, { error: msg });
    }
  });
  return server;
}

export function startServer(
  opts: ServerOptions,
  listen: { port?: number; host?: string } = {},
): Promise<{ url: string; port: number; close: () => void }> {
  const port = listen.port ?? 8787;
  const host = listen.host ?? '127.0.0.1';
  const server = createServer(opts);
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      const addr = server.address();
      const realPort = typeof addr === 'object' && addr !== null ? addr.port : port;
      resolve({
        url: `http://${host}:${realPort}`,
        port: realPort,
        close: () => server.close(),
      });
    });
  });
}

import * as fs from 'node:fs';
import * as path from 'node:path';
import { VectorStore, type StoreOptions } from './store.js';
import { startServer } from './server.js';

function flag(argv: string[], name: string): string | undefined {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : undefined;
}

function storeOptionsFrom(flags: { data?: string; dim?: string; metric?: string; M?: string; ef?: string }): StoreOptions {
  return {
    dataDir: flags.data ?? './skipverse-data',
    dim: Number(flags.dim ?? 64),
    metric: (flags.metric as StoreOptions['metric']) ?? 'cosine',
    M: flags.M ? Number(flags.M) : undefined,
    efConstruction: flags.ef ? Number(flags.ef) : undefined,
  };
}

const USAGE = `skipverse — a from-scratch vector database (HNSW, zero deps)

  skipverse serve   --port 8787 --data ./data --dim 64 --metric cosine
                    [--M 16] [--ef-construction 200] [--quantization sq8|sq4]
  skipverse calibrate --data ./data --dim 64
                    (freeze ranges + rewrite codes on a quantized store)
  skipverse import  --file vectors.jsonl --data ./data --dim 64
                    (JSONL: {"id": "...", "vec": [...]})
  skipverse help
`;

async function main(): Promise<void> {
  const [cmd, ...argv] = process.argv.slice(2);
  switch (cmd) {
    case undefined:
    case 'help':
    case '--help':
    case '-h': {
      console.log(USAGE);
      return;
    }
    case 'serve': {
      const port = Number(flag(argv, '--port') ?? 8787);
      const so = storeOptionsFrom({
        data: flag(argv, '--data'),
        dim: flag(argv, '--dim'),
        metric: flag(argv, '--metric'),
        M: flag(argv, '--M'),
        ef: flag(argv, '--ef-construction'),
      });
      const store = VectorStore.open(so);
      const demoDir = path.resolve(import.meta.dirname, '../../demo');
      const { url } = await startServer({ store, demoDir: fs.existsSync(demoDir) ? demoDir : undefined }, { port });
      console.log(`skipverse serving ${so.dim}d ${so.metric} index (${store.index.size} vectors)`);
      console.log(`  api    ${url}/search  POST {vec, k, ef}`);
      console.log(`  demo   ${url}/`);
      console.log(`  data   ${so.dataDir}`);
      process.on('SIGINT', () => {
        console.log('\nshutting down (checkpoint=false — WAL has everything)');
        store.close();
        process.exit(0);
      });
      break;
    }
    case 'calibrate': {
      const so = storeOptionsFrom({
        data: flag(argv, '--data'),
        dim: flag(argv, '--dim'),
        metric: flag(argv, '--metric'),
        M: flag(argv, '--M'),
        ef: flag(argv, '--ef-construction'),
      });
      const store = VectorStore.open(so);
      const t0 = performance.now();
      store.calibrate();
      store.close();
      console.log(
        `calibrated ${store.index.size} vectors in ${Math.round(performance.now() - t0)}ms → ${store.index.bytesPerVector} B/vec`,
      );
      break;
    }
    case 'import': {
      const file = flag(argv, '--file');
      if (!file) throw new Error('import requires --file <vectors.jsonl>');
      const so = storeOptionsFrom({
        data: flag(argv, '--data'),
        dim: flag(argv, '--dim'),
        metric: flag(argv, '--metric'),
        M: flag(argv, '--M'),
        ef: flag(argv, '--ef-construction'),
      });
      const store = VectorStore.open(so);
      const t0 = performance.now();
      let n = 0;
      let batch: { id: string; vec: number[] }[] = [];
      const flush = () => {
        store.upsertBatch(batch);
        n += batch.length;
        batch = [];
      };
      const lines = fs.readFileSync(file, 'utf8').split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.length === 0) continue;
        batch.push(JSON.parse(trimmed) as { id: string; vec: number[] });
        if (batch.length >= 500) flush();
      }
      if (batch.length > 0) flush();
      store.checkpoint();
      store.close();
      const ms = Math.round(performance.now() - t0);
      console.log(`imported ${n} vectors in ${ms}ms → ${so.dataDir}`);
      break;
    }
    default:
      throw new Error(`unknown command: ${cmd}\n${USAGE}`);
  }
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});

import { build } from 'esbuild';

await build({
  entryPoints: ['demo/app.ts'],
  bundle: true,
  outfile: 'demo/bundle.js',
  format: 'iife',
  target: 'es2020',
  minify: true,
  logLevel: 'info',
});

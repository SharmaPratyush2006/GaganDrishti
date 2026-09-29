#!/usr/bin/env node
/**
 * Export a Phase 6 product's nDSM as .glb (uncompressed and Draco), headless.
 *
 * In plain words: this is the viewer's "Export glTF" button without the
 * browser. It loads the product with the viewer's own loader (load.js,
 * products.js, over the data route's own path resolution) and writes the files
 * with the viewer's own exporter (gltfExport.js). No geometry code lives here.
 *
 *   node viewer/scripts/export-mesh.mjs --phase6-report data/outputs/demo/processed/phase6/synthetic/phase6_report.json \
 *        --out data/outputs/demo/mesh [--product <name>] [--outputs-dir <repo>/data/outputs]
 *
 * Writes the .glb files and mesh_summary.json into --out. Exit 1 on failure.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import draco3d from 'draco3dgltf';

import { loadDemo } from '../src/data/load.js';
import { exportTerrainGlb } from '../src/export/gltfExport.js';
import { diskFetch } from '../server/diskFetch.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
  options: {
    'phase6-report': { type: 'string' },
    product: { type: 'string' },
    out: { type: 'string' },
    'outputs-dir': { type: 'string', default: path.resolve(here, '..', '..', 'data', 'outputs') },
  },
});

async function main() {
  if (!args['phase6-report'] || !args.out) throw new Error('--phase6-report and --out are required');
  const fetchImpl = diskFetch(path.resolve(args['outputs-dir']));
  const demo = await loadDemo({ phase6Report: args['phase6-report'], productName: args.product }, fetchImpl);
  const raster = demo.rasters.ndsm;
  if (!raster) throw new Error(`nDSM not available: ${demo.notAvailable.ndsm ?? 'unknown reason'}`);
  const sourceImage = demo.alignment.image ? demo.rasters.image : null;
  const encoder = await draco3d.createEncoderModule();
  fs.mkdirSync(args.out, { recursive: true });

  const files = [];
  for (const draco of [false, true]) {
    const result = await exportTerrainGlb({ raster, sourceImage, draco, dracoEncoder: draco ? encoder : null });
    const file = path.join(args.out, result.fileName);
    fs.writeFileSync(file, result.bytes);
    files.push({
      path: path.relative(process.cwd(), file).split(path.sep).join('/'),
      bytes: result.stats.byteLength,
      vertices: result.stats.vertices,
      triangles: result.stats.triangles,
      draco_requested: draco,
      draco_verified: result.stats.draco.verified,
      mode: result.meta.mode,
      units: result.meta.units,
      height_values: result.meta.heightValues,
      texture_included: result.meta.texture.included,
    });
  }
  const summary = {
    exporter: 'viewer/src/export/gltfExport.js',
    phase6_report: args['phase6-report'],
    product: demo.products.productName,
    surface: 'ndsm',
    files,
  };
  fs.writeFileSync(path.join(args.out, 'mesh_summary.json'), JSON.stringify(summary, null, 2));
  process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
}

main().catch((err) => {
  process.stderr.write(`export-mesh failed: ${err.message}\n`);
  process.exit(1);
});

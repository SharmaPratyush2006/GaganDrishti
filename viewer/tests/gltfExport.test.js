/**
 * Step 7: glTF (.glb) export, with and without Draco.
 * Every GLB is parsed: by an independent header/chunk parser (inspectGlb) and
 * read back with gltf-transform; Draco files are decoded with Google's
 * official decoder (draco3dgltf) and compared with the raster.
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

import { WebIO } from '@gltf-transform/core';
import { KHRDracoMeshCompression } from '@gltf-transform/extensions';
import draco3d from 'draco3dgltf';
import { beforeAll, describe, expect, it } from 'vitest';

import { heightmapFromRaster } from '../src/3d/heightmap.js';
import { loadDracoEncoder } from '../src/export/dracoEncoder.js';
import {
  buildExportMesh, DRACO_EXTENSION, DRACO_QUANTIZATION, ExportError, exportFileName, exportTerrainGlb, inspectGlb,
} from '../src/export/gltfExport.js';
import { encodePng } from '../src/export/png.js';
import { loadDemo } from '../src/data/load.js';
import { makeRaster, mapToPixel, pixelToMap, validityMask } from '../src/data/raster.js';
import { diskFetch, expected, hasPhase6Outputs, PHASE6_REPORT, REPO_ROOT } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

const T = expected['tiny_ndsm_cog.tif'].transform;
const ABS = { transform: T, crs: 'EPSG:32643', units: 'metres' };
const pattern = (r, c) => ((r * 7 + c * 13) % 17) + 0.25;
const ROTATED = [0.3, -0.4, 700000, 0.4, 0.3, 3170000];

let encoder;
let decoder;
beforeAll(async () => {
  encoder = await draco3d.createEncoderModule();
  decoder = await draco3d.createDecoderModule();
});

/** Read a GLB back with gltf-transform (Draco decoded when present). */
async function readBack(bytes) {
  const io = new WebIO().registerExtensions([KHRDracoMeshCompression]).registerDependencies({ 'draco3d.decoder': decoder });
  const doc = await io.readBinary(bytes);
  const prim = doc.getRoot().listMeshes()[0].listPrimitives()[0];
  return {
    doc,
    positions: prim.getAttribute('POSITION').getArray(),
    normals: prim.getAttribute('NORMAL').getArray(),
    uvs: prim.getAttribute('TEXCOORD_0').getArray(),
    indices: prim.getIndices().getArray(),
    material: prim.getMaterial(),
  };
}

/** Pixel (row, col) and continuous pixel coordinate of an exported vertex, via the transform. */
function vertexPixel(hm, positions, v) {
  const [c, r] = mapToPixel(hm.transform, positions[v * 3] + hm.origin[0], -positions[v * 3 + 2] + hm.origin[1]);
  return { colF: c, rowF: r, row: Math.floor(r), col: Math.floor(c) };
}

/** Independent count of triangles with 3 valid vertices (the viewer's triangulation at stride 1). */
function expectedTriangles(raster) {
  const ok = (r, c) => raster.valid[r * raster.width + c] === 1;
  let n = 0;
  for (let r = 0; r < raster.height - 1; r++) {
    for (let c = 0; c < raster.width - 1; c++) {
      if (ok(r, c) && ok(r + 1, c) && ok(r, c + 1)) n++;
      if (ok(r, c + 1) && ok(r + 1, c) && ok(r + 1, c + 1)) n++;
    }
  }
  return n;
}

/** Minimal PNG reader (8-bit, filter 0 rows, as png.js writes) using Node's zlib. */
function readPng(bytes) {
  const b = Buffer.from(bytes);
  expect([...b.subarray(0, 8)]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  let o = 8;
  let ihdr;
  const idat = [];
  while (o < b.length) {
    const len = b.readUInt32BE(o);
    const type = b.toString('latin1', o + 4, o + 8);
    const data = b.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') ihdr = { width: data.readUInt32BE(0), height: data.readUInt32BE(4), depth: data[8], color: data[9] };
    if (type === 'IDAT') idat.push(data);
    o += 12 + len;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const ch = ihdr.color === 0 ? 1 : 4;
  const stride = ihdr.width * ch;
  const px = new Uint8Array(stride * ihdr.height);
  for (let r = 0; r < ihdr.height; r++) {
    expect(raw[r * (stride + 1)]).toBe(0);
    px.set(raw.subarray(r * (stride + 1) + 1, (r + 1) * (stride + 1)), r * stride);
  }
  return { ...ihdr, channels: ch, px };
}

function grayImage(h, w, f, transform = T) {
  const n = h * w;
  const values = new Uint8Array(n);
  for (let i = 0; i < n; i++) values[i] = f(Math.floor(i / w), i % w);
  return makeRaster({ width: w, height: h, values, valid: validityMask(values, null), dtype: 'uint8', crs: 'EPSG:32643', transform,
    productType: 'image', source: 'test-image', bands: 1, bandDtypes: ['uint8'], photometric: 1, pixels: values });
}

describe('valid GLB', () => {
  it('writes a structurally valid glTF 2.0 binary with one indexed terrain primitive', async () => {
    const raster = rasterFrom(grid(9, 11, pattern), ABS);
    const out = await exportTerrainGlb({ raster });
    const ins = inspectGlb(out.bytes);
    expect(ins.version).toBe(2);
    expect(ins.length).toBe(out.bytes.byteLength);
    expect(ins.json.asset.version).toBe('2.0');
    expect(ins.json.asset.generator).toMatch(/DepthWizard/);
    expect(ins.primitives).toBe(1);
    expect(ins.binLength).toBeGreaterThan(0);
    const p = ins.json.meshes[0].primitives[0];
    expect(Object.keys(p.attributes).sort()).toEqual(['NORMAL', 'POSITION', 'TEXCOORD_0']);
    expect(ins.json.accessors[p.indices].componentType).toBe(5125); // UNSIGNED_INT
    expect(ins.json.accessors[p.attributes.POSITION].count).toBe(99);
    expect(ins.json.accessors[p.indices].count).toBe(3 * 2 * 8 * 10);
    expect(ins.draco.used).toBe(false);
    expect(out.stats.draco).toEqual({ requested: false, verified: false });
    // And gltf-transform reads exactly what was built.
    const back = await readBack(out.bytes);
    const mesh = buildExportMesh(raster);
    expect(Array.from(back.positions)).toEqual(Array.from(mesh.tile.positions));
    expect(Array.from(back.indices)).toEqual(Array.from(mesh.tile.indices));
    expect(Array.from(back.normals)).toEqual(Array.from(mesh.normals));
  });

  it('raster dimensions: one vertex per valid pixel, 2 (W-1)(H-1) triangles when fully valid', async () => {
    const raster = rasterFrom(grid(12, 20, pattern), ABS);
    const out = await exportTerrainGlb({ raster });
    expect(out.stats.vertices).toBe(240);
    expect(out.stats.triangles).toBe(2 * 11 * 19);
    expect(out.meta).toMatchObject({ width: 20, height: 12 });
    expect(out.fileName).toBe('test_ndsm_ABSOLUTE.glb');
  });
});

describe('no raster / invalid data / failures', () => {
  it('refuses to export without a raster or a height product', async () => {
    await expect(exportTerrainGlb({ raster: null })).rejects.toThrow(ExportError);
    await expect(exportTerrainGlb({ raster: null })).rejects.toThrow(/no raster is loaded/);
    const conf = rasterFrom(grid(3, 3, () => 1), { ...ABS, productType: 'confidence' });
    await expect(exportTerrainGlb({ raster: conf })).rejects.toThrow(/not a height product/);
  });

  it('refuses an all-nodata raster (nothing invented)', async () => {
    await expect(exportTerrainGlb({ raster: rasterFrom(grid(4, 4, () => null), ABS) })).rejects.toThrow(/no valid triangle/);
  });

  it('Draco requested without an encoder fails instead of writing an uncompressed file', async () => {
    const raster = rasterFrom(grid(4, 4, pattern), ABS);
    await expect(exportTerrainGlb({ raster, draco: true, dracoEncoder: null })).rejects.toThrow(/no Draco encoder is available; nothing was exported/);
  });

  it('a broken encoder surfaces as a rejected export, not a mislabelled file', async () => {
    const raster = rasterFrom(grid(4, 4, pattern), ABS);
    await expect(exportTerrainGlb({ raster, draco: true, dracoEncoder: {} })).rejects.toThrow();
  });

  it('inspectGlb rejects non-GLB bytes and truncated files', async () => {
    expect(() => inspectGlb(new Uint8Array(40))).toThrow(/magic/);
    expect(() => inspectGlb(new Uint8Array(4))).toThrow(/shorter/);
    const good = (await exportTerrainGlb({ raster: rasterFrom(grid(3, 3, pattern), ABS) })).bytes;
    expect(() => inspectGlb(good.slice(0, good.length - 4))).toThrow(/length/);
  });

  it('Draco availability: the browser loader reports "unavailable" with a reason where it cannot fetch its WASM (Node)', async () => {
    const r = await loadDracoEncoder();
    expect(r.ok).toBe(false);
    expect(typeof r.reason).toBe('string');
    expect(r.reason.length).toBeGreaterThan(0);
  });
});

describe('geometry semantics', () => {
  it('pixel centres through the affine transform (north-up and rotated); UVs are texel centres', async () => {
    for (const transform of [T, ROTATED]) {
      const raster = rasterFrom(grid(7, 9, pattern), { ...ABS, transform });
      const hm = heightmapFromRaster(raster, { verticalScale: 1 });
      const back = await readBack((await exportTerrainGlb({ raster })).bytes);
      const seen = new Set();
      for (let v = 0; v < back.positions.length / 3; v++) {
        const p = vertexPixel(hm, back.positions, v);
        expect(p.colF).toBeCloseTo(p.col + 0.5, 5);
        expect(p.rowF).toBeCloseTo(p.row + 0.5, 5);
        expect(back.positions[v * 3 + 1]).toBe(raster.values[p.row * 9 + p.col]);
        expect(back.uvs[v * 2]).toBeCloseTo((p.col + 0.5) / 9, 6);
        expect(back.uvs[v * 2 + 1]).toBeCloseTo((p.row + 0.5) / 7, 6);
        const [mx, my] = pixelToMap(transform, p.col + 0.5, p.row + 0.5);
        expect(back.positions[v * 3] + hm.origin[0]).toBeCloseTo(mx, 4);
        expect(-back.positions[v * 3 + 2] + hm.origin[1]).toBeCloseTo(my, 4);
        seen.add(p.row * 9 + p.col);
      }
      expect(seen.size).toBe(63);
    }
  });

  it('triangles face up (+y normals) for north-up and mirrored grids', async () => {
    for (const transform of [T, [0.5, 0, 700000, 0, 0.5, 3170000]]) {
      const raster = rasterFrom(grid(5, 5, () => 3), { ...ABS, transform });
      const back = await readBack((await exportTerrainGlb({ raster })).bytes);
      for (let v = 0; v < back.normals.length / 3; v++) expect(back.normals[v * 3 + 1]).toBeCloseTo(1, 6);
    }
  });

  it('nodata: no vertex at a nodata pixel, only fully valid triangles, counts recorded', async () => {
    const raster = rasterFrom(grid(8, 8, (r, c) => ((r === 3 && c === 4) || (r === 0 && c === 7) ? null : pattern(r, c))), ABS);
    const hm = heightmapFromRaster(raster, { verticalScale: 1 });
    const out = await exportTerrainGlb({ raster });
    const back = await readBack(out.bytes);
    const pixels = new Set();
    for (let v = 0; v < back.positions.length / 3; v++) {
      const p = vertexPixel(hm, back.positions, v);
      pixels.add(`${p.row},${p.col}`);
      expect(Number.isFinite(back.positions[v * 3 + 1])).toBe(true);
      expect(back.positions[v * 3 + 1]).not.toBe(-9999);
    }
    expect(pixels.has('3,4')).toBe(false);
    expect(pixels.has('0,7')).toBe(false);
    expect(out.stats.vertices).toBe(62);
    expect(out.stats.triangles).toBe(expectedTriangles(raster));
    expect(out.meta.nodata).toMatchObject({ value: -9999, invalidPixels: 2, trianglesOmitted: 2 * 7 * 7 - expectedTriangles(raster) });
  });

  it('ABSOLUTE: y is the raster value in metres, 1:1', async () => {
    const raster = rasterFrom(grid(4, 4, (r, c) => 540 + pattern(r, c)), { ...ABS, productType: 'dsm' });
    const out = await exportTerrainGlb({ raster });
    const back = await readBack(out.bytes);
    const ys = new Set(Array.from({ length: back.positions.length / 3 }, (_, v) => back.positions[v * 3 + 1]));
    expect(ys).toEqual(new Set(Array.from(raster.values)));
    expect(out.meta).toMatchObject({ mode: 'ABSOLUTE', units: 'metres', verticalScaleApplied: 1, displayedVerticalScale: 1, crs: 'EPSG:32643' });
    expect(out.meta.heightValues).toBe('y = raster value in metres (ABSOLUTE)');
  });

  it('RELATIVE: raw unitless values, display exaggeration recorded but not applied, never metres', async () => {
    const raster = rasterFrom(grid(6, 6, (r, c) => (r + c) / 20), { productType: 'relative_height' });
    const out = await exportTerrainGlb({ raster });
    const back = await readBack(out.bytes);
    const ys = Array.from({ length: back.positions.length / 3 }, (_, v) => back.positions[v * 3 + 1]);
    expect(Math.max(...ys)).toBe(Math.fround(0.5));
    expect(out.meta.mode).toBe('RELATIVE');
    expect(out.meta.verticalScaleApplied).toBe(1);
    expect(out.meta.displayedVerticalScale).toBeGreaterThan(1);
    expect(out.meta.displayedVerticalScaleReason).toMatch(/display-only exaggeration/);
    expect(out.meta.heightValues).toBe('y = raster value, RELATIVE and unitless — NOT metres');
    expect(out.meta.units).toBeNull();
    expect(out.meta.crs).toBeNull();
    expect(out.meta.horizontalUnits).toMatch(/pixels/);
    expect(out.fileName).toMatch(/RELATIVE\.glb$/);
  });
});

describe('metadata / extras', () => {
  it('asset.extras and node.extras identify source, CRS, size, units, mode, scale, nodata, frame', async () => {
    const raster = rasterFrom(grid(5, 6, pattern), { ...ABS, transform: ROTATED });
    raster.synthetic = true;
    const ins = inspectGlb((await exportTerrainGlb({ raster })).bytes);
    const meta = ins.json.asset.extras.depthwizard;
    expect(ins.json.nodes[0].extras.depthwizard).toEqual(meta);
    expect(meta).toMatchObject({
      source: 'test', productType: 'ndsm', synthetic: true, width: 6, height: 5, crs: 'EPSG:32643', transform: ROTATED,
      units: 'metres', mode: 'ABSOLUTE', verticalScaleApplied: 1,
      nodata: { value: -9999, invalidPixels: 0 },
      compression: { draco: false },
    });
    expect(meta.coordinateFrame.axes).toMatch(/-z = map north/);
    expect(meta.coordinateFrame.origin).toEqual(heightmapFromRaster(raster).origin);
    expect(meta.notIncluded.join()).toMatch(/overlays.*uncertainty/);
    expect(JSON.stringify(meta)).not.toMatch(/probabilit|±|sigma/i);
  });
});

describe('texture', () => {
  it('an aligned grayscale source image is embedded losslessly as a 1-channel PNG', async () => {
    const raster = rasterFrom(grid(4, 6, pattern), ABS);
    const img = grayImage(4, 6, (r, c) => r * 40 + c);
    const out = await exportTerrainGlb({ raster, sourceImage: img });
    expect(out.meta.texture).toMatchObject({ included: true, png: 'grayscale', content: 'grayscale source image (not RGB)' });
    const back = await readBack(out.bytes);
    const tex = back.material.getBaseColorTexture();
    expect(tex.getMimeType()).toBe('image/png');
    const png = readPng(tex.getImage());
    expect([png.width, png.height, png.depth, png.color]).toEqual([6, 4, 8, 0]);
    expect(Array.from(png.px)).toEqual(Array.from(img.values));
  });

  it('a misaligned source image is not embedded; the reason is recorded', async () => {
    const raster = rasterFrom(grid(4, 6, pattern), ABS);
    const img = grayImage(4, 6, () => 9, [0.5, 0, 700010, 0, -0.5, 3170000]);
    const out = await exportTerrainGlb({ raster, sourceImage: img });
    expect(out.meta.texture.included).toBe(false);
    expect(out.meta.texture.reason).toMatch(/not aligned/);
    expect(inspectGlb(out.bytes).json.images).toBeUndefined();
  });

  it('png.js RGBA round-trips exactly', async () => {
    const data = Uint8Array.from({ length: 3 * 2 * 4 }, (_, i) => (i * 37) % 256);
    const png = readPng(await encodePng({ width: 3, height: 2, data, format: 'rgba' }));
    expect([png.width, png.height, png.color]).toEqual([3, 2, 6]);
    expect(Array.from(png.px)).toEqual(Array.from(data));
  });
});

describe('Draco (KHR_draco_mesh_compression)', () => {
  it('the written GLB is really Draco-compressed, and decodes to the same mesh within the quantisation step', async () => {
    const raster = rasterFrom(grid(40, 50, (r, c) => (r > 10 && r < 20 && c > 5 && c < 25 ? 18 : 0.5) + pattern(r, c) / 100), { ...ABS, transform: ROTATED });
    const plain = await exportTerrainGlb({ raster });
    const out = await exportTerrainGlb({ raster, draco: true, dracoEncoder: encoder });
    const ins = inspectGlb(out.bytes);
    expect(ins.draco).toMatchObject({ used: true, required: true, compressedPrimitives: 1 });
    expect(out.stats.draco).toEqual({ requested: true, verified: true });
    expect(ins.json.meshes[0].primitives[0].extensions[DRACO_EXTENSION].attributes).toHaveProperty('POSITION');
    expect(out.meta.compression).toEqual({ draco: true, extension: DRACO_EXTENSION, lossy: true, quantizationBits: DRACO_QUANTIZATION });
    expect(out.bytes.byteLength).toBeLessThan(plain.bytes.byteLength);

    const back = await readBack(out.bytes);
    const hm = heightmapFromRaster(raster, { verticalScale: 1 });
    expect(back.positions.length / 3).toBe(plain.stats.vertices);
    expect(back.indices.length / 3).toBe(plain.stats.triangles);
    // Every decoded vertex lies within the 16-bit quantisation step of its pixel's exact sample.
    const mesh = buildExportMesh(raster);
    const p = mesh.tile.positions;
    const range = [0, 1, 2].map((k) => { let lo = Infinity; let hi = -Infinity; for (let i = k; i < p.length; i += 3) { lo = Math.min(lo, p[i]); hi = Math.max(hi, p[i]); } return hi - lo; });
    const step = Math.max(...range) / (2 ** DRACO_QUANTIZATION.POSITION - 1);
    let maxErr = 0;
    const pixels = new Set();
    for (let v = 0; v < back.positions.length / 3; v++) {
      const px = vertexPixel(hm, back.positions, v);
      const exact = [...pixelToMap(ROTATED, px.col + 0.5, px.row + 0.5)];
      const ex = [exact[0] - hm.origin[0], raster.values[px.row * 50 + px.col], -(exact[1] - hm.origin[1])];
      for (let k = 0; k < 3; k++) maxErr = Math.max(maxErr, Math.abs(back.positions[v * 3 + k] - ex[k]));
      pixels.add(px.row * 50 + px.col);
    }
    expect(pixels.size).toBe(plain.stats.vertices); // one-to-one with the valid pixels
    expect(maxErr).toBeLessThanOrEqual(step);
    console.log(`[draco fixture] ${plain.bytes.byteLength} B uncompressed -> ${out.bytes.byteLength} B Draco; max position error ${maxErr.toExponential(2)} (step ${step.toExponential(2)})`);
  });
});

describe.skipIf(!hasPhase6Outputs())('real Phase 6 nDSM', () => {
  it('uncompressed + Draco export of the acceptance nDSM: dimensions, heights at Phase 6 clicks, Draco verified', async () => {
    const d = await loadDemo({ phase6Report: PHASE6_REPORT }, diskFetch());
    const raster = d.rasters.ndsm;
    const report = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, d.products.phase6Report), 'utf-8'));
    const plain = await exportTerrainGlb({ raster, sourceImage: d.rasters.image });
    let valid = 0;
    for (let i = 0; i < raster.valid.length; i++) valid += raster.valid[i];
    expect(plain.stats.vertices).toBe(valid);
    expect(plain.stats.triangles).toBe(expectedTriangles(raster));
    expect(plain.meta.texture).toMatchObject({ included: true, png: 'grayscale' });
    const back = await readBack(plain.bytes);
    const hm = heightmapFromRaster(raster, { verticalScale: 1 });
    const yAt = new Map();
    for (let v = 0; v < back.positions.length / 3; v++) {
      const px = vertexPixel(hm, back.positions, v);
      yAt.set(px.row * raster.width + px.col, back.positions[v * 3 + 1]);
    }
    const [a, , c, , e, f] = raster.transform;
    for (const click of report.products[d.products.productName].clicks) {
      const col = Math.floor((click.easting_m - c) / a);
      const row = Math.floor((click.northing_m - f) / e);
      expect(yAt.get(row * raster.width + col)).toBe(click.ndsm_m);
    }
    const dr = await exportTerrainGlb({ raster, sourceImage: d.rasters.image, draco: true, dracoEncoder: encoder });
    expect(inspectGlb(dr.bytes).draco).toMatchObject({ used: true, required: true, compressedPrimitives: 1 });
    expect(dr.bytes.byteLength).toBeLessThan(plain.bytes.byteLength);
    console.log(`[real nDSM] ${raster.width}x${raster.height}: ${plain.stats.vertices} vertices, ${plain.stats.triangles} triangles; `
      + `${plain.bytes.byteLength} B uncompressed (${plain.stats.exportMs.toFixed(0)} ms) -> ${dr.bytes.byteLength} B Draco (${dr.stats.exportMs.toFixed(0)} ms), Node`);
  }, 60000);
});

it('exportFileName is filesystem-safe', () => {
  expect(exportFileName({ source: '/data/outputs/phase6/x y/ndsm.tif', productType: 'ndsm', mode: 'ABSOLUTE' }, true)).toBe('ndsm_ABSOLUTE_draco.glb');
  // A name that already ends with the product type does not repeat it.
  expect(exportFileName({ source: '/data/outputs/phase3/JAX_1_relative_height.npy', productType: 'relative_height', mode: 'RELATIVE' }, false))
    .toBe('JAX_1_relative_height_RELATIVE.glb');
  expect(exportFileName({ source: 'x/tile 7.tif', productType: 'dsm', mode: 'ABSOLUTE' }, false)).toBe('tile_7_dsm_ABSOLUTE.glb');
});

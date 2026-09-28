import path from 'node:path';

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { heightmapFromRaster, samplePosition, worldToMap, worldYToValue } from '../src/3d/heightmap.js';
import { buildQuadtree } from '../src/3d/quadtree.js';
import { tileGeometry } from '../src/3d/terrain.js';
import { buildTileMesh } from '../src/3d/tileMesh.js';
import { readGeoTiff } from '../src/data/geotiffLoader.js';
import { formatValue } from '../src/data/metadata.js';
import { readNpyRaster } from '../src/data/npy.js';
import { valueAt } from '../src/data/raster.js';
import { expected, FIXTURES, readArrayBuffer } from './helpers.js';
import { covers, grid, rasterFrom, trianglesRC } from './mesh3d.helpers.js';

const FIXTURE_T = expected['tiny_ndsm_cog.tif'].transform; // from the Step 2 fixture, not typed in
const ABS = { transform: FIXTURE_T, crs: expected['tiny_ndsm_cog.tif'].crs, units: 'metres' };

/** Whole raster as one full-resolution tile. */
function fullRes(raster) {
  const hm = heightmapFromRaster(raster);
  const root = buildQuadtree(raster.width, raster.height, { tileSize: 1024 });
  expect(root.stride).toBe(1);
  return { hm, tile: buildTileMesh(hm, root) };
}

describe('geometry', () => {
  it('maps raster dimensions to a W x H vertex grid with 2 triangles per cell', () => {
    const { tile } = fullRes(rasterFrom(grid(3, 4, (r, c) => r + c), ABS));
    expect(tile.vertexCount).toBe(12);
    expect(tile.triangleCount).toBe(2 * 3 * 2);
    expect(tile.indices).toBeInstanceOf(Uint32Array); // indexed, vertices shared
  });

  it('places vertices from the raster transform and keeps heights unchanged (1:1 for ABSOLUTE)', async () => {
    const raster = await readGeoTiff(readArrayBuffer(path.join(FIXTURES, 'tiny_ndsm_cog.tif')), { source: 'fixture' });
    const { hm, tile } = fullRes(raster);
    expect(hm.verticalScale).toBe(1);
    const [a, , c, , e, f] = raster.transform;
    for (let v = 0; v < tile.vertexCount; v++) {
      const p = tile.pixelIndex[v];
      const row = Math.floor(p / raster.width);
      const col = p % raster.width;
      const [x, y, z] = tile.positions.subarray(3 * v, 3 * v + 3);
      // map = world + origin: the pixel centre through the file's own transform
      const [mx, my] = worldToMap(hm, x, z);
      expect(mx).toBeCloseTo(c + a * (col + 0.5), 6);
      expect(my).toBeCloseTo(f + e * (row + 0.5), 6);
      expect(y).toBe(valueAt(raster, row, col)); // exact: the raster's own float32 value
      expect(worldYToValue(hm, y)).toBe(valueAt(raster, row, col));
    }
  });

  it('horizontal spacing follows the transform (pixel size and orientation)', () => {
    const { hm } = fullRes(rasterFrom(grid(2, 2, () => 1), ABS));
    const [x00, , z00] = samplePosition(hm, 0, 0);
    const [x01, , z01] = samplePosition(hm, 0, 1);
    const [x10, , z10] = samplePosition(hm, 1, 0);
    expect(x01 - x00).toBeCloseTo(FIXTURE_T[0], 12); // one column east = a
    expect(z01 - z00).toBeCloseTo(0, 12);
    expect(z10 - z00).toBeCloseTo(-FIXTURE_T[4], 12); // one row south = -e (world z is south)
    expect(x10 - x00).toBeCloseTo(0, 12);
  });

  it('follows a rotated transform instead of assuming north-up', () => {
    const rot = [0.3, -0.4, 1000, 0.4, 0.3, 2000]; // rotated, and rows point north
    const { hm } = fullRes(rasterFrom(grid(2, 2, () => 1), { ...ABS, transform: rot }));
    const [x00, , z00] = samplePosition(hm, 0, 0);
    const [x01, , z01] = samplePosition(hm, 0, 1);
    expect(x01 - x00).toBeCloseTo(0.3, 12);
    expect(z01 - z00).toBeCloseTo(-0.4, 12);
  });

  it('generates upward normals, for north-up and for mirrored (south-up) transforms', () => {
    for (const t of [FIXTURE_T, [0.5, 0, 700000, 0, 0.5, 3170000]]) {
      const { tile } = fullRes(rasterFrom(grid(3, 3, () => 7), { ...ABS, transform: t }));
      const g = tileGeometry(tile);
      const n = g.getAttribute('normal');
      expect(n.count).toBe(tile.vertexCount);
      for (let i = 0; i < n.count; i++) expect(n.getY(i)).toBeCloseTo(1, 6);
    }
  });

  it('builds an indexed THREE.BufferGeometry with the raster index per vertex', () => {
    const { tile } = fullRes(rasterFrom(grid(3, 3, (r, c) => r * c), ABS));
    const g = tileGeometry(tile);
    expect(g).toBeInstanceOf(THREE.BufferGeometry);
    expect(g.index.count).toBe(tile.triangleCount * 3);
    expect(g.getAttribute('pixelIndex').count).toBe(tile.vertexCount);
  });
});

describe('nodata', () => {
  it('1. all-valid raster: every cell gives two triangles', () => {
    const { tile } = fullRes(rasterFrom(grid(3, 3, () => 1), ABS));
    expect(tile.triangleCount).toBe(8);
    expect(tile.droppedForNodata).toBe(0);
  });

  it('2. one invalid interior vertex removes exactly the triangles that use it', () => {
    const { tile } = fullRes(rasterFrom(grid(3, 3, (r, c) => (r === 1 && c === 1 ? null : 1)), ABS));
    // The centre sample is shared by 6 of the 8 triangles (fixed diagonal).
    expect(tile.triangleCount).toBe(2);
    expect(tile.droppedForNodata).toBe(6);
    expect(Array.from(tile.pixelIndex)).not.toContain(1 * 3 + 1);
  });

  it('3. an invalid corner removes only the one triangle that uses it', () => {
    const { tile } = fullRes(rasterFrom(grid(3, 3, (r, c) => (r === 0 && c === 0 ? null : 1)), ABS));
    expect(tile.triangleCount).toBe(7);
    expect(Array.from(tile.pixelIndex)).not.toContain(0);
  });

  it('4. a contiguous nodata region stays a hole', () => {
    const hole = (r, c) => r >= 2 && r <= 3 && c >= 2 && c <= 4;
    const raster = rasterFrom(grid(7, 8, (r, c) => (hole(r, c) ? null : 3)), ABS);
    const { tile } = fullRes(raster);
    const tris = trianglesRC(tile, raster.width);
    for (const t of tris) for (const [r, c] of t) expect(hole(r, c)).toBe(false);
    // Nothing covers the inside of the hole.
    for (const [r, c] of [[2.5, 2.5], [2.5, 3.5], [3, 3], [2, 4]]) expect(tris.some((t) => covers(t, r, c))).toBe(false);
    // Exactly the triangles whose three corners are valid remain.
    let expectedCount = 0;
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 7; c++) {
        const ok = (rr, cc) => !hole(rr, cc);
        if (ok(r, c) && ok(r + 1, c) && ok(r, c + 1)) expectedCount++;
        if (ok(r, c + 1) && ok(r + 1, c) && ok(r + 1, c + 1)) expectedCount++;
      }
    }
    expect(tile.triangleCount).toBe(expectedCount);
  });

  it('5. zero is a valid height and nodata never becomes zero', () => {
    const raster = rasterFrom([[0, 0, 0], [0, 5, null]], ABS);
    const { tile } = fullRes(raster);
    const ys = [];
    for (let v = 0; v < tile.vertexCount; v++) ys.push(tile.positions[3 * v + 1]);
    expect(ys.filter((y) => y === 0).length).toBe(4); // the four real zeros
    expect(ys).not.toContain(-9999);
    expect(Array.from(tile.pixelIndex)).not.toContain(5); // the nodata sample is not a vertex
    expect(tile.triangleCount).toBe(3); // only the triangle touching nodata is gone
  });

  it('6. negative valid heights are kept as they are', () => {
    const raster = rasterFrom(grid(2, 3, (r, c) => -5 - r - c), ABS);
    const { tile } = fullRes(raster);
    const ys = new Set();
    for (let v = 0; v < tile.vertexCount; v++) ys.add(tile.positions[3 * v + 1]);
    expect([...ys].sort((a, b) => a - b)).toEqual([-8, -7, -6, -5]);
  });

  it('coarse tiles never bridge nodata they skip over', () => {
    const bad = [3, 5];
    const raster = rasterFrom(grid(9, 9, (r, c) => (r === bad[0] && c === bad[1] ? null : 2)), ABS);
    const hm = heightmapFromRaster(raster);
    const root = buildQuadtree(9, 9, { tileSize: 4 });
    expect(root.stride).toBe(2); // samples rows/cols 0,2,4,6,8 -- the bad pixel is NOT a vertex
    const tile = buildTileMesh(hm, root);
    const tris = trianglesRC(tile, 9);
    expect(tris.some((t) => covers(t, bad[0], bad[1]))).toBe(false);
    expect(tile.droppedForNodata).toBeGreaterThan(0);
  });
});

describe('ABSOLUTE / RELATIVE', () => {
  it('an ABSOLUTE raster stays ABSOLUTE, metric, 1:1', () => {
    const { hm } = fullRes(rasterFrom(grid(2, 2, () => 540), ABS));
    expect(hm.raster.mode).toBe('ABSOLUTE');
    expect(hm.verticalScale).toBe(1);
    expect(hm.horizontalUnits).toMatch(/EPSG:32643 map units/);
  });

  it('a RELATIVE .npy stays RELATIVE, pixel-space, with an explicit display-only scale', () => {
    // numpy.save layout, built inline (2x3 float32)
    const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    const padded = header + ' '.repeat(64 - ((10 + header.length + 1) % 64)) + '\n';
    const bytes = new Uint8Array(10 + padded.length + 24);
    bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
    new DataView(bytes.buffer).setUint16(8, padded.length, true);
    bytes.set(new TextEncoder().encode(padded), 10);
    bytes.set(new Uint8Array(new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).buffer), 10 + padded.length);
    const raster = readNpyRaster(bytes.buffer);
    const { hm, tile } = fullRes(raster);
    expect(hm.raster.mode).toBe('RELATIVE');
    expect(hm.georeferenced).toBe(false);
    expect(hm.horizontalUnits).toMatch(/pixels/);
    expect(hm.verticalScale).not.toBe(1);
    expect(hm.verticalScaleReason).toMatch(/display-only/);
    // The raw relative value is recoverable from the rendered height.
    const y = tile.positions[3 * 0 + 1];
    expect(worldYToValue(hm, y)).toBeCloseTo(valueAt(raster, 0, 0), 6);
    expect(formatValue(worldYToValue(hm, y), raster)).not.toMatch(/\bm\b|metre|meter/);
  });

  it('refuses to mesh a product that is not a height', () => {
    const conf = rasterFrom(grid(2, 2, () => 3), { ...ABS, productType: 'confidence', units: null });
    expect(() => heightmapFromRaster(conf)).toThrow(/not a height product/);
  });
});

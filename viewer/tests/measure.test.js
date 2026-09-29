import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { heightmapFromRaster, sampleXZ } from '../src/3d/heightmap.js';
import { createMeasureTool, measureAtWorldPoint, NO_VALID_HEIGHT, raycastTerrain } from '../src/3d/measure.js';
import { initialCameraPose } from '../src/3d/scene.js';
import { TerrainLOD } from '../src/3d/terrain.js';
import { loadDemo } from '../src/data/load.js';
import { readNpyRaster } from '../src/data/npy.js';
import { pixelToMap } from '../src/data/raster.js';
import { domEvent, fakeCanvas } from './dom.helpers.js';
import { diskFetch, expected, hasPhase6Outputs, PHASE6_REPORT } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

const T = expected['tiny_ndsm_cog.tif'].transform;
const ABS = { transform: T, crs: 'EPSG:32643', units: 'metres' };
const pattern = (r, c) => ((r * 7 + c * 13) % 17) + 0.25; // every pixel different

/** A camera straight above world (x, z), and the canvas-centre client point. */
function lookDownAt(x, z, height = 500) {
  const camera = new THREE.PerspectiveCamera(20, 800 / 500, 0.1, 1e5);
  camera.up.set(0, 0, -1);
  camera.position.set(x, height, z);
  camera.lookAt(x, 0, z);
  camera.updateMatrixWorld();
  return { camera, client: [400, 250] };
}

function world(hm, row, col, dRow = 0, dCol = 0) {
  // Map coordinate of a point (dRow, dCol) pixels from the centre of (row, col), then to world.
  const [mx, my] = pixelToMap(hm.transform, col + 0.5 + dCol, row + 0.5 + dRow);
  return { x: mx - hm.origin[0], y: 0, z: -(my - hm.origin[1]) };
}

describe('raycast -> pixel -> raster value', () => {
  const raster = rasterFrom(grid(33, 33, pattern), ABS);
  const hm = heightmapFromRaster(raster);
  const terrain = new TerrainLOD(hm, { tileSize: 64 });
  terrain.update([0, 1e4, 0]);

  it('14-16. the ray hits the terrain; the hit maps to the right pixel; the value is the raster value', () => {
    for (const [row, col] of [[3, 4], [16, 16], [30, 2]]) {
      const [x, z] = sampleXZ(hm, row, col);
      const { camera, client } = lookDownAt(x, z);
      const hit = raycastTerrain({ camera, domElement: fakeCanvas(), terrain }, ...client);
      expect(hit).not.toBeNull();
      const m = measureAtWorldPoint(hit.point, { heightmap: hm, measureRaster: raster });
      expect(m.ok).toBe(true);
      expect([m.row, m.col]).toEqual([row, col]);
      expect(m.value).toBe(raster.values[row * 33 + col]);
    }
  });

  it('17. pixel-centre convention: a point stays in its pixel up to the pixel edge', () => {
    for (const [dr, dc, want] of [[0, 0, [10, 10]], [0.49, 0.49, [10, 10]], [-0.49, -0.49, [10, 10]], [0.51, 0, [11, 10]], [0, -0.51, [10, 9]]]) {
      const m = measureAtWorldPoint(world(hm, 10, 10, dr, dc), { heightmap: hm, measureRaster: raster });
      expect([m.row, m.col]).toEqual(want);
    }
  });

  it('18. the raster transform is respected (rotated grid)', () => {
    const rot = rasterFrom(grid(9, 9, pattern), { ...ABS, transform: [0.3, -0.4, 700000, 0.4, 0.3, 3170000] });
    const h = heightmapFromRaster(rot);
    for (const [row, col] of [[0, 0], [2, 7], [8, 8]]) {
      const [x, z] = sampleXZ(h, row, col);
      const m = measureAtWorldPoint({ x, y: 0, z }, { heightmap: h, measureRaster: rot });
      expect([m.row, m.col, m.value]).toEqual([row, col, rot.values[row * 9 + col]]);
      expect(m.map[0]).toBeCloseTo(pixelToMap(rot.transform, col + 0.5, row + 0.5)[0], 6);
    }
  });

  it('19. a nodata pixel is rejected', () => {
    const r = rasterFrom(grid(9, 9, (row, col) => (row === 4 && col === 4 ? null : 1)), ABS);
    const h = heightmapFromRaster(r);
    const m = measureAtWorldPoint(world(h, 4, 4), { heightmap: h, measureRaster: r });
    expect(m.ok).toBe(false);
    expect(m.message).toBe(NO_VALID_HEIGHT);
    expect(m.reason).toMatch(/nodata/);
    expect(m.value).toBeUndefined();
  });

  it('20. a point outside the raster is rejected', () => {
    const m = measureAtWorldPoint(world(hm, -3, 5), { heightmap: hm, measureRaster: raster });
    expect(m.ok).toBe(false);
    expect(m.reason).toMatch(/outside the raster/);
  });

  it('21. a non-finite value is rejected even if marked valid', () => {
    const r = rasterFrom(grid(5, 5, () => 2), ABS);
    r.values[2 * 5 + 2] = NaN;
    r.valid[2 * 5 + 2] = 1; // defensive path: bypass the loader's own mask
    const h = heightmapFromRaster(rasterFrom(grid(5, 5, () => 2), ABS));
    const m = measureAtWorldPoint(world(h, 2, 2), { heightmap: h, measureRaster: r });
    expect(m.ok).toBe(false);
    expect(m.reason).toMatch(/non-finite/);
  });

  it('22. ABSOLUTE nDSM reports metres', () => {
    const m = measureAtWorldPoint(world(hm, 5, 5), { heightmap: hm, measureRaster: raster });
    expect(m.mode).toBe('ABSOLUTE');
    expect(m.units).toBe('metres');
    expect(m.formatted).toBe(`${raster.values[5 * 33 + 5].toFixed(2)} m`);
    expect(m.crs).toBe('EPSG:32643');
  });

  it('23. RELATIVE .npy stays unitless', () => {
    const vals = Float32Array.from({ length: 16 }, (_, i) => i / 10);
    const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (4, 4), }";
    const padded = header + ' '.repeat(64 - ((10 + header.length + 1) % 64)) + '\n';
    const bytes = new Uint8Array(10 + padded.length + vals.byteLength);
    bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
    new DataView(bytes.buffer).setUint16(8, padded.length, true);
    bytes.set(new TextEncoder().encode(padded), 10);
    bytes.set(new Uint8Array(vals.buffer), 10 + padded.length);
    const rel = readNpyRaster(bytes.buffer);
    const h = heightmapFromRaster(rel);
    const m = measureAtWorldPoint(world(h, 2, 3), { heightmap: h, measureRaster: rel });
    expect(m.ok).toBe(true);
    expect(m.value).toBe(rel.values[2 * 4 + 3]);
    expect(m.mode).toBe('RELATIVE');
    expect(m.units).toBe('unitless (relative)');
    expect(m.formatted).not.toMatch(/\bm\b|metre|meter/);
    expect(m.map).toBeNull();
  });

  it('24. measuring never modifies the raster', () => {
    const before = Float32Array.from(raster.values);
    for (let i = 0; i < 50; i++) measureAtWorldPoint(world(hm, i % 33, (i * 7) % 33), { heightmap: hm, measureRaster: raster });
    expect(raster.values).toEqual(before);
  });

  it('27. no uncertainty is produced', () => {
    const m = measureAtWorldPoint(world(hm, 5, 5), { heightmap: hm, measureRaster: raster });
    expect(JSON.stringify(m)).not.toMatch(/uncertain|sigma|confidence|±|std/i);
  });
});

describe('the reported value comes from the source raster, not the mesh', () => {
  it('25. a coarse LOD tile: the value is the raster pixel, not the (interpolated) mesh height', () => {
    // Checkerboard: a stride-2 tile only has even-index samples as vertices, all 0;
    // odd pixels hold 50. The coarse mesh is flat at 0 there, the raster is not.
    const r = rasterFrom(grid(17, 17, (row, col) => (row % 2 && col % 2 ? 50 : 0)), ABS);
    const hm = heightmapFromRaster(r);
    const terrain = new TerrainLOD(hm, { tileSize: 8 });
    terrain.update([0, 1e5, 0]); // far: the coarse root only
    expect(terrain.group.children.map((m) => m.userData.stride)).toEqual([2]);
    const [x, z] = sampleXZ(hm, 5, 7);
    const { camera, client } = lookDownAt(x, z);
    const hit = raycastTerrain({ camera, domElement: fakeCanvas(), terrain }, ...client);
    expect(hit.tile.stride).toBe(2);
    expect(hit.point.y).toBeCloseTo(0, 6); // what the mesh shows
    const m = measureAtWorldPoint(hit.point, { heightmap: hm, measureRaster: r });
    expect([m.row, m.col]).toEqual([5, 7]);
    expect(m.value).toBe(50); // what the raster holds
    // Same pixel at full resolution gives the same value.
    terrain.update([x, 1, z]);
    const fine = raycastTerrain({ camera, domElement: fakeCanvas(), terrain }, ...client);
    expect(fine.tile.stride).toBe(1);
    expect(measureAtWorldPoint(fine.point, { heightmap: hm, measureRaster: r }).value).toBe(50);
  });

  it('a DSM surface is shown, the nDSM is measured (different values at the same pixel)', () => {
    const dsm = rasterFrom(grid(9, 9, (row, col) => 540 + pattern(row, col)), { ...ABS, productType: 'dsm' });
    const ndsm = rasterFrom(grid(9, 9, pattern), ABS);
    const hm = heightmapFromRaster(dsm);
    const terrain = new TerrainLOD(hm);
    terrain.update([0, 1e4, 0]);
    const [x, z] = sampleXZ(hm, 3, 6);
    const { camera, client } = lookDownAt(x, z, 2000);
    const hit = raycastTerrain({ camera, domElement: fakeCanvas(), terrain }, ...client);
    const m = measureAtWorldPoint(hit.point, { heightmap: hm, measureRaster: ndsm });
    expect(m.productType).toBe('ndsm');
    expect(m.value).toBe(ndsm.values[3 * 9 + 6]);
    expect(m.value).not.toBeCloseTo(hit.point.y, 1);
    expect(m.displayed.productType).toBe('dsm');
    expect(m.displayed.value).toBe(dsm.values[3 * 9 + 6]);
  });

  it('a different CRS cannot be related and is rejected', () => {
    const ndsm = rasterFrom(grid(9, 9, pattern), { ...ABS, crs: 'EPSG:32644' });
    const hm = heightmapFromRaster(rasterFrom(grid(9, 9, pattern), ABS));
    const m = measureAtWorldPoint(world(hm, 1, 1), { heightmap: hm, measureRaster: ndsm });
    expect(m.ok).toBe(false);
    expect(m.reason).toMatch(/CRS differs/);
  });
});

describe('measure tool', () => {
  function tool() {
    const raster = rasterFrom(grid(33, 33, pattern), ABS);
    const hm = heightmapFromRaster(raster);
    const terrain = new TerrainLOD(hm);
    terrain.update([0, 1e4, 0]);
    const [x, z] = sampleXZ(hm, 12, 20);
    const { camera, client } = lookDownAt(x, z, 50); // ~0.035 m per screen pixel
    const scene = new THREE.Scene();
    const el = fakeCanvas();
    const results = [];
    const t = createMeasureTool({ camera, domElement: el, terrain, scene, measureRaster: raster, render: () => {}, onMeasure: (r) => results.push(r) });
    return { raster, hm, terrain, camera, client, scene, el, t, results };
  }

  it('26. repeated clicks replace the measurement and move the single marker', () => {
    const s = tool();
    s.t.setActive(true);
    const a = s.t.clickAt(...s.client);
    const b = s.t.clickAt(s.client[0] + 60, s.client[1] - 40); // ~4 px east, ~3 px north
    expect(s.t.latest).toBe(b);
    expect(b.ok).toBe(true);
    expect([b.row, b.col]).not.toEqual([a.row, a.col]);
    expect(s.scene.children.filter((o) => o.name === 'measure-marker').length).toBe(1);
    expect(s.t.marker.position.x).toBeCloseTo(b.world[0], 9);
  });

  it('a click (press+release in place) measures; a drag does not; inactive does nothing', () => {
    const s = tool();
    const click = (dx) => {
      s.el.dispatchEvent(domEvent('pointerdown', { button: 0, clientX: 400, clientY: 250 }));
      s.el.dispatchEvent(domEvent('pointerup', { button: 0, clientX: 400 + dx, clientY: 250 }));
    };
    click(0);
    expect(s.results.length).toBe(0);
    s.t.setActive(true);
    click(20);
    expect(s.results.length).toBe(0);
    click(1);
    expect(s.results.length).toBe(1);
    expect([s.results[0].row, s.results[0].col]).toEqual([12, 20]);
    s.t.setActive(false); // exit clears the result and hides the marker
    expect(s.t.latest).toBeNull();
    expect(s.t.marker.visible).toBe(false);
  });

  it('a click that misses the terrain reports no valid height', () => {
    const s = tool();
    s.camera.lookAt(s.camera.position.x, s.camera.position.y + 10, s.camera.position.z - 1); // look at the sky
    s.camera.updateMatrixWorld();
    const r = s.t.clickAt(...s.client);
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/did not hit the terrain/);
  });
});

describe.skipIf(!hasPhase6Outputs())('real Phase 6 nDSM', () => {
  it('clicking each building centroid returns the Phase 6 height_at() nDSM value (from the report)', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const { REPO_ROOT } = await import('./helpers.js');
    const d = await loadDemo({ phase6Report: PHASE6_REPORT }, diskFetch());
    const report = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, d.products.phase6Report), 'utf-8'));
    for (const surface of ['ndsm', 'dsm']) {
      const hm = heightmapFromRaster(d.rasters[surface]);
      const terrain = new TerrainLOD(hm);
      terrain.update(initialCameraPose(terrain.bounds()).position);
      for (const click of report.products[d.products.productName].clicks) {
        const x = click.easting_m - hm.origin[0];
        const z = -(click.northing_m - hm.origin[1]);
        const { camera, client } = lookDownAt(x, z, 2000);
        const hit = raycastTerrain({ camera, domElement: fakeCanvas(), terrain }, ...client);
        const m = measureAtWorldPoint(hit.point, { heightmap: hm, measureRaster: d.rasters.ndsm });
        expect(m.ok).toBe(true);
        expect(m.value).toBe(click.ndsm_m);
        expect(m.units).toBe('metres');
      }
      terrain.dispose();
    }
  });
});

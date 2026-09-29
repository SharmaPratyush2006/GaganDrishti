import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { heightmapFromRaster, sampleXZ } from '../src/3d/heightmap.js';
import { initialCameraPose } from '../src/3d/scene.js';
import { TerrainLOD } from '../src/3d/terrain.js';
import { loadDemo } from '../src/data/load.js';
import { readNpyRaster } from '../src/data/npy.js';
import { valueAt } from '../src/data/raster.js';
import { buildIndex } from '../server/dataRoute.js';
import { diskFetch, expected, hasPhase6Outputs, OUTPUTS_DIR, PHASE6_REPORT, readArrayBuffer, REPO_ROOT } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

const ABS = { transform: expected['tiny_ndsm_cog.tif'].transform, crs: 'EPSG:32643', units: 'metres' };

/** Every rendered vertex height equals the raster value at that vertex's pixel. */
function checkVerticesAgainstRaster(terrain, raster) {
  let checked = 0;
  for (const mesh of terrain.group.children) {
    const pos = mesh.geometry.getAttribute('position');
    const pix = mesh.geometry.getAttribute('pixelIndex');
    for (let i = 0; i < pos.count; i++) {
      const p = pix.getX(i);
      const v = valueAt(raster, Math.floor(p / raster.width), p % raster.width);
      // Positions are float32: exact for scale 1, float32-rounded product otherwise.
      if (v === null || pos.getY(i) !== Math.fround(v * terrain.heightmap.verticalScale)) return { ok: false, at: p, v, y: pos.getY(i) };
      checked++;
    }
  }
  return { ok: true, checked };
}

describe('TerrainLOD', () => {
  const raster = rasterFrom(grid(129, 129, (r, c) => (r > 40 && r < 60 && c > 40 && c < 60 ? 30 : 1)), ABS);
  const hm = heightmapFromRaster(raster);

  it('re-uses cached tile geometry instead of rebuilding when the LOD repeats', () => {
    const t = new TerrainLOD(hm, { tileSize: 8 });
    const near = [...sampleXZ(hm, 0, 0)];
    const cam = [near[0], 5, near[1]];
    const first = t.update(cam);
    expect(first.tilesBuiltThisUpdate).toBe(first.selectedTiles);
    const again = t.update(cam);
    expect(again.tilesBuiltThisUpdate).toBe(0);
    expect(t.group.children.length).toBe(again.selectedTiles);
    const far = t.update([0, 10000, 10000]);
    expect(far.triangles).toBeLessThan(first.triangles);
    const back = t.update(cam);
    expect(back.tilesBuiltThisUpdate).toBe(0);
    t.dispose();
  });

  it('every rendered vertex carries the raster value at its pixel', () => {
    const t = new TerrainLOD(hm, { tileSize: 8 });
    t.update([0, 20, 40]);
    expect(checkVerticesAgainstRaster(t, raster).ok).toBe(true);
    t.dispose();
  });
});

const available = hasPhase6Outputs();

describe.skipIf(!available)('real Phase 6 surfaces through one mesh pipeline', () => {
  it('meshes DSM, DTM and nDSM with the same code, heights straight from the raster', async () => {
    const d = await loadDemo({ phase6Report: PHASE6_REPORT }, diskFetch());
    const summary = {};
    for (const type of ['dsm', 'dtm', 'ndsm']) {
      const raster = d.rasters[type];
      const hm = heightmapFromRaster(raster);
      const t = new TerrainLOD(hm);
      const pose = initialCameraPose(t.bounds());
      const stats = t.update(pose.position);
      expect(hm.raster.mode).toBe('ABSOLUTE');
      expect(hm.verticalScale).toBe(1);
      expect(stats.triangles).toBeGreaterThan(0);
      expect(Object.keys(stats.tilesByLevel).length).toBeGreaterThan(1); // LOD mixes levels
      const check = checkVerticesAgainstRaster(t, raster);
      expect(check.ok).toBe(true);
      const b = t.bounds();
      expect(b.max[1] - b.min[1]).toBeGreaterThan(0); // not flat
      summary[type] = { triangles: stats.triangles, vertices: stats.vertices, ymin: b.min[1], ymax: b.max[1] };
      t.dispose();
    }
    // nDSM heights are above-ground, DSM heights are absolute elevations.
    expect(summary.ndsm.ymax).toBeLessThan(summary.dsm.ymin);
  });

  it('renders a Phase 3 .npy through the same pipeline, still RELATIVE', () => {
    const npyPath = buildIndex(OUTPUTS_DIR).npy.find((p) => p.includes('relative_height'));
    const raster = readNpyRaster(readArrayBuffer(path.join(REPO_ROOT, npyPath)), { source: npyPath });
    const hm = heightmapFromRaster(raster);
    const t = new TerrainLOD(hm);
    const stats = t.update(initialCameraPose(t.bounds()).position);
    expect(hm.raster.mode).toBe('RELATIVE');
    expect(hm.horizontalUnits).toMatch(/pixels/);
    expect(stats.triangles).toBeGreaterThan(0);
    expect(checkVerticesAgainstRaster(t, raster).ok).toBe(true);
    t.dispose();
  });
});

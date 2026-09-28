/**
 * End-to-end on the real Phase 4b/5/6 SYNTHETIC outputs, through the same
 * index + path resolution the dev server uses. Expected values are read from
 * the pipeline's own report (height_at() results), never typed in here.
 * Skipped when the git-ignored outputs are absent.
 */
import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { loadDemo } from '../src/data/load.js';
import { mapToPixelIndex, validStats, valueAt } from '../src/data/raster.js';
import { buildIndex } from '../server/dataRoute.js';
import { diskFetch, hasPhase6Outputs, OUTPUTS_DIR, REPO_ROOT } from './helpers.js';

const available = hasPhase6Outputs();

describe.skipIf(!available)('real Phase 6 SYNTHETIC outputs', () => {
  const fetchImpl = diskFetch();
  // The same index the loader reads decides which report is used.
  const reportPath = available ? buildIndex(OUTPUTS_DIR).reports.find((p) => p.endsWith('phase6_report.json')) : null;
  const report = reportPath ? JSON.parse(fs.readFileSync(path.join(REPO_ROOT, reportPath), 'utf-8')) : null;

  it('loads the acceptance product with preserved size, nodata and georeferencing', async () => {
    const d = await loadDemo({}, fetchImpl);
    const name = report.acceptance.input;
    const grid = report.products[name].grid;
    expect(d.products.productName).toBe(name);
    for (const type of ['dsm', 'dtm', 'ndsm']) {
      const r = d.rasters[type];
      expect(r.productType).toBe(type);
      expect([r.width, r.height]).toEqual([grid.width, grid.height]);
      expect(r.crs).toBe(grid.crs);
      expect(r.transform).toEqual(grid.transform);
      expect(r.mode).toBe('ABSOLUTE');
      const exp = report.products[name].exports[`${type}.tif`];
      expect(r.nodata).toBe(exp.nodata);
      // No valid pixel carries the nodata value.
      let validNodata = 0;
      for (let i = 0; i < r.values.length; i++) if (r.valid[i] && r.values[i] === r.nodata) validNodata++;
      expect(validNodata).toBe(0);
    }
    expect(d.rasters.ground_mask.nodata).toBe(report.products[name].exports['ground_mask.tif'].nodata);
    expect(d.rasters.image.mode).toBeNull();
    expect(d.alignment).toMatchObject({ dsm: true, dtm: true, ground_mask: true, image: true });
  });

  it('looks up nDSM (not DSM) at each building centroid and matches Phase 6 height_at()', async () => {
    for (const name of Object.keys(report.products)) {
      const d = await loadDemo({ productName: name }, fetchImpl);
      for (const click of report.products[name].clicks) {
        const px = mapToPixelIndex(d.rasters.ndsm, click.easting_m, click.northing_m);
        expect(px).not.toBeNull();
        const ndsm = valueAt(d.rasters.ndsm, px.row, px.col);
        const dsm = valueAt(d.rasters.dsm, px.row, px.col);
        expect(ndsm).toBe(click.ndsm_m);
        expect(dsm).toBe(click.dsm_m);
        expect(ndsm).not.toBe(dsm); // DSM elevation is not building height
      }
    }
  });

  it('links Phase 5 layers only where Phase 5 validated that exact source DSM', async () => {
    for (const name of Object.keys(report.products)) {
      const d = await loadDemo({ productName: name }, fetchImpl);
      const has = Boolean(d.products.errorMap);
      expect(Boolean(d.rasters.error)).toBe(has);
      if (has) {
        expect(d.rasters.error.productType).toBe('error');
        expect(d.rasters.error.tags.PRODUCT).toBe('dsm');
        expect(d.alignment.error).toBe(true);
        expect(d.rasters.confidence.productType).toBe('confidence');
        expect(validStats(d.rasters.error).nodataPixels).toBeGreaterThanOrEqual(0);
      } else {
        expect(d.notAvailable.error).toBeTruthy();
      }
      expect(d.notAvailable.uncertainty).toMatch(/no per-pixel uncertainty/);
    }
  });
});

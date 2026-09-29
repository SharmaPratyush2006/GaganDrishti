/**
 * Phase 8: the demo manifest and .npy choices are explicit, and what the demo
 * loads keeps its units (ABSOLUTE nDSM in metres; RELATIVE .npy unitless).
 */
import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { chooseNpy, isDemoManifest, loadPhase8Demo, readDemoManifest } from '../src/data/demo.js';
import { fetchNpy, loadDemo } from '../src/data/load.js';
import { formatValue } from '../src/data/metadata.js';
import { resolveDemoProducts } from '../src/data/products.js';
import { mapToPixelIndex, valueAt } from '../src/data/raster.js';
import { buildIndex } from '../server/dataRoute.js';
import { diskFetch, OUTPUTS_DIR, PHASE6_REPORT, REPO_ROOT } from './helpers.js';

const DEMO_MANIFEST = 'data/outputs/demo/reports/demo_report.json';

/** A fetch() over an in-memory index and JSON files (url -> body). */
function memoryFetch(index, files = {}) {
  const respond = (status, body) => ({ ok: status === 200, status, json: async () => body });
  return async (url) => {
    if (url === '/__data_index') return respond(200, { root: 'data/outputs', npy: [], ...index });
    return url in files ? respond(200, files[url]) : respond(404, 'not found');
  };
}

describe('Phase 8 selection rules (no index-order guessing)', () => {
  it('recognises only the demo manifest', () => {
    expect(isDemoManifest(DEMO_MANIFEST)).toBe(true);
    expect(isDemoManifest('data/outputs/demo/processed/phase6/synthetic/phase6_report.json')).toBe(false);
    expect(isDemoManifest('data/outputs/demo/reports/not_demo_report.json.bak')).toBe(false);
  });

  it('chooses a .npy only when chosen or unique', () => {
    const one = { npy: ['data/outputs/a.npy'] };
    const two = { npy: ['data/outputs/a.npy', 'data/outputs/b.npy'] };
    expect(chooseNpy(one)).toBe('data/outputs/a.npy');
    expect(chooseNpy(two, 'data/outputs/b.npy')).toBe('data/outputs/b.npy');
    expect(() => chooseNpy(two)).toThrow(/several .npy relative-height outputs found; choose one/);
    expect(() => chooseNpy(two, 'data/outputs/c.npy')).toThrow(/not found under data\/outputs/);
    expect(() => chooseNpy({ npy: [] })).toThrow(/no .npy/);
  });

  it('refuses several Phase 6 reports without a choice, and loads the chosen one', () => {
    const r6 = (dir) => ({
      path: `data/outputs/${dir}/phase6_report.json`,
      json: { acceptance: { input: 'x.tif' }, products: { 'x.tif': { exports: { 'ndsm.tif': { path: `data/outputs/${dir}/x/ndsm.tif` } } } } },
    });
    const reports = [r6('phase6/synthetic'), r6('demo/processed/phase6/synthetic')];
    expect(() => resolveDemoProducts(reports)).toThrow(/several Phase 6 reports found; choose one/);
    const p = resolveDemoProducts(reports, { phase6Report: 'data/outputs/demo/processed/phase6/synthetic/phase6_report.json' });
    expect(p.surfaces.ndsm.path).toBe('data/outputs/demo/processed/phase6/synthetic/x/ndsm.tif');
  });

  it('reads the manifest and the Phase 6 choice it records', async () => {
    const manifest = { viewer: { phase6_report: 'data/outputs/demo/p6/phase6_report.json' }, phase6_surfaces: { acceptance_input: 'x.tif' } };
    const got = await readDemoManifest(memoryFetch({ reports: [DEMO_MANIFEST] }, { [`/${DEMO_MANIFEST}`]: manifest }));
    expect(got).toMatchObject({ path: DEMO_MANIFEST, phase6Report: manifest.viewer.phase6_report, productName: 'x.tif' });
  });

  it('refuses a missing, duplicated or incomplete manifest with the reason', async () => {
    await expect(readDemoManifest(memoryFetch({ reports: [] }))).rejects.toThrow(/no Phase 8 demo found .*depthwizard.demo synthetic/);
    const two = [DEMO_MANIFEST, 'data/outputs/demo2/reports/demo_report.json'];
    await expect(readDemoManifest(memoryFetch({ reports: two }))).rejects.toThrow(/several Phase 8 demo manifests found/);
    await expect(readDemoManifest(memoryFetch({ reports: [DEMO_MANIFEST] }))).rejects.toThrow(/HTTP 404/);
    const noLink = memoryFetch({ reports: [DEMO_MANIFEST] }, { [`/${DEMO_MANIFEST}`]: { phase: '8' } });
    await expect(readDemoManifest(noLink)).rejects.toThrow(/records no viewer.phase6_report/);
  });
});

// Real, git-ignored outputs of `python -m depthwizard.demo synthetic`; skipped when absent.
const index = buildIndex(OUTPUTS_DIR);
const hasDemo = index.reports.includes(DEMO_MANIFEST);

describe.skipIf(!hasDemo)('real Phase 8 demo outputs', () => {
  const fetchImpl = diskFetch();
  const manifest = hasDemo ? JSON.parse(fs.readFileSync(path.join(REPO_ROOT, DEMO_MANIFEST), 'utf-8')) : null;

  it('loads exactly the Phase 6 report and product the manifest names, in metres', async () => {
    const d = await loadPhase8Demo(fetchImpl);
    expect(d.demo.path).toBe(DEMO_MANIFEST);
    expect(d.products.phase6Report).toBe(manifest.viewer.phase6_report);
    expect(d.products.productName).toBe(manifest.phase6_surfaces.acceptance_input);
    expect(d.products.synthetic).toBe(true);
    for (const type of ['dsm', 'dtm', 'ndsm']) {
      expect(d.rasters[type].mode).toBe('ABSOLUTE');
      expect(formatValue(1.5, d.rasters[type])).toBe('1.50 m');
    }
    expect(d.products.notAvailable.uncertainty).toMatch(/no per-pixel uncertainty/);
  });

  it('reads the manifest nDSM click values from the raster it loads', async () => {
    const d = await loadPhase8Demo(fetchImpl);
    for (const c of manifest.phase6_surfaces.clicks) {
      const px = mapToPixelIndex(d.rasters.ndsm, c.easting_m, c.northing_m);
      expect(px).not.toBeNull();
      expect(valueAt(d.rasters.ndsm, px.row, px.col)).toBe(c.ndsm_m);
    }
  });

  it('keeps the overlay rule: nothing on the acceptance nDSM, the demo Phase 5 maps on synthetic_dsm.tif', async () => {
    const d = await loadPhase8Demo(fetchImpl);
    expect(d.products.errorMap).toBeNull();
    expect(d.products.notAvailable.error).toMatch(/no Phase 5 report validated/);
    const sec = await loadDemo({ phase6Report: manifest.viewer.phase6_report, productName: 'synthetic_dsm.tif' }, fetchImpl);
    // The demo's own Phase 5 run, never the separate data/outputs/phase5 run of another DSM.
    expect(sec.products.errorMap.path.startsWith('data/outputs/demo/validation/')).toBe(true);
    expect(sec.products.confidence.path.startsWith('data/outputs/demo/validation/')).toBe(true);
  });

  it.skipIf(!index.reports.includes(PHASE6_REPORT))('leaves the Phase 7 run selectable, and refuses to guess between the two', async () => {
    await expect(loadDemo({}, fetchImpl)).rejects.toThrow(/several Phase 6 reports found; choose one/);
    const p7 = await loadDemo({ phase6Report: PHASE6_REPORT }, fetchImpl);
    expect(p7.products.phase6Report).toBe(PHASE6_REPORT);
  });
});

const uploadNpy = index.npy.find((p) => p.startsWith('data/outputs/uploads/'));

describe.skipIf(!uploadNpy)('a /process relative field in the viewer', () => {
  it('is RELATIVE and never printed in metres', async () => {
    const r = await fetchNpy(chooseNpy(index, uploadNpy), diskFetch());
    expect(r.mode).toBe('RELATIVE');
    expect(formatValue(1.5, r)).toBe('1.5000 (unitless, relative)');
    expect(formatValue(1.5, r)).not.toMatch(/ m$/);
  });
});

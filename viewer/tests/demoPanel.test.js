/**
 * Phase 8 DemoPanel: shows the manifest as recorded, labels everything
 * SYNTHETIC, and writes "not yet measured" instead of any missing number.
 */
import fs from 'node:fs';
import path from 'node:path';

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import DemoPanel from '../src/components/DemoPanel.jsx';
import { REPO_ROOT } from './helpers.js';

const render = (manifest, p = 'data/outputs/demo/reports/demo_report.json') =>
  renderToStaticMarkup(createElement(DemoPanel, { manifest, path: p }))
    .replace(/<[^>]+>/g, ' ').replace(/&#x27;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ');

/** The manifest's shape (demo/synthetic.py), with values chosen to test the rules. */
function manifest(overrides = {}) {
  return {
    label: 'SYNTHETIC: heights were SPECIFIED, not measured.',
    mode: 'ABSOLUTE',
    units: 'metres',
    input: {
      image: 'data/outputs/demo/input/synthetic_city_a.tif',
      routing: { reason: 'georeferenced raster with CRS, geotransform and both sun angles' },
      metadata: { width_px: 512, height_px: 512, band_count: 1, dtype: 'uint8', crs: 'EPSG:32643', gsd_m: 0.5,
        sun_elevation_deg: 45, sun_azimuth_deg: 135, sun_elevation_source: 'gdal:default[SUN_ELEVATION_DEG]=45.0',
        sun_azimuth_source: 'gdal:default[SUN_AZIMUTH_DEG]=135.0' },
    },
    phase2_shadow_physics: {
      footprints: 'SUPPLIED from the fixture\'s truth sidecar',
      buildings: [
        { building: 'tower_a', shadow_length_px: 59.397, shadow_length_m: 29.6985, tan_elevation: 1, height_m: 29.6985,
          specified_height_m: 30, abs_error_m: 0.3015, confidence: 'nominal' },
        { building: 'lost_e', shadow_length_px: null, shadow_length_m: null, tan_elevation: 1, height_m: null,
          specified_height_m: 9, abs_error_m: null, confidence: 'failed' },
      ],
      succeeded: 1, mae_m: 0.25, max_abs_error_m: 0.3, pixel_grid_bound_m: 0.3536,
      error_source: 'pixel quantisation', figure: null,
    },
    phase5_validation: {
      products: [
        { product: 'dsm', units: 'metres', overall_mae: 0.314, overall_rmse: 0.3156, building_mae: 0.162, error_map_figure: null },
        { product: 'rel', units: 'unitless', overall_mae: 0.5, overall_rmse: null, building_mae: 0.25, error_map_figure: null },
      ],
      real_world_validation: 'not yet measured (this run is SYNTHETIC)',
    },
    phase6_surfaces: { clicks: [{ building: 'tower_a', specified_height_m: 30, ndsm_m: 29.99997901916504, abs_error_m: 2.1e-5 }] },
    viewer: { overlays: 'Phase 5 overlays validate the Phase 4b DSM' },
    uncertainty: 'not yet measured',
    not_yet_measured: { fps: 'not yet measured', per_pixel_uncertainty: 'not yet measured' },
    ...overrides,
  };
}

describe('DemoPanel', () => {
  const text = render(manifest());

  it('shows the input metadata with its provenance', () => {
    expect(text).toContain('ABSOLUTE (heights in metres)');
    expect(text).toContain('EPSG:32643');
    expect(text).toContain('0.500 m');
    expect(text).toContain('45.000° ← gdal:default[SUN_ELEVATION_DEG]=45.0');
    expect(text).toContain('135.000° ← gdal:default[SUN_AZIMUTH_DEG]=135.0');
  });

  it('labels the shadow physics and validation as synthetic, never as real-world', () => {
    expect(text).toContain('SYNTHETIC: heights were SPECIFIED, not measured.');
    expect(text).toContain('Shadow Physics — Synthetic Fixture (Phase 2)');
    expect(text).toContain('SPECIFIED heights (synthetic ground truth), not real-world measurements');
    expect(text).toContain('Validation (Phase 5, synthetic)');
    expect(text).toContain('Real-world validation: not yet measured (this run is SYNTHETIC)');
    expect(text).not.toMatch(/zero-shot|unseen/i);
  });

  it('shows a failed measurement and a missing metric as "not yet measured", never 0', () => {
    const row = /lost_e (.*?) failed/.exec(text)[1];
    expect(row).toContain('not yet measured');
    expect(row).not.toMatch(/0\.000/);
    expect(text).toContain('RMSE not yet measured');
    expect(text).toContain('Uncertainty: not yet measured');
    expect(text).toContain('fps: not yet measured');
    expect(text).toContain('per pixel uncertainty: not yet measured');
  });

  it('puts metres only on metre-valued products', () => {
    expect(text).toContain('DSM vs specified truth: overall MAE 0.314 m, RMSE 0.316 m, building MAE 0.162 m');
    // Unitless product: no " m" after any of its values (the next item follows directly).
    expect(text).toContain('REL vs specified truth: overall MAE 0.500, RMSE not yet measured, building MAE 0.250 Real-world validation');
  });

  it('never appends " m" to a missing metric of a metre-valued product', () => {
    const m = manifest();
    m.phase5_validation.products[0].overall_rmse = null;
    const t = render(m);
    expect(t).toContain('DSM vs specified truth: overall MAE 0.314 m, RMSE not yet measured, building MAE 0.162 m');
    expect(t).not.toContain('not yet measured m');
  });

  it('reports the nDSM click values and Phase 2 summary as recorded', () => {
    expect(text).toContain('tower_a 30.0 30.0000 0.000021');
    expect(text).toContain('SYNTHETIC MAE 0.250 m, max 0.300 m (1 of 2 buildings)');
  });
});

const REAL = path.join(REPO_ROOT, 'data', 'outputs', 'demo', 'reports', 'demo_report.json');

describe.skipIf(!fs.existsSync(REAL))('DemoPanel on the real demo manifest', () => {
  const real = JSON.parse(fs.readFileSync(REAL, 'utf-8'));
  const text = render(real);

  it('shows every Phase 2 building and the recorded MAE / max', () => {
    for (const b of real.phase2_shadow_physics.buildings) {
      expect(text).toContain(`${b.building} ${b.shadow_length_px.toFixed(2)} ${b.shadow_length_m.toFixed(3)}`);
      expect(text).toContain(b.height_m.toFixed(3));
    }
    expect(text).toContain(`SYNTHETIC MAE ${real.phase2_shadow_physics.mae_m.toFixed(3)} m, max ${real.phase2_shadow_physics.max_abs_error_m.toFixed(3)} m`);
  });

  it('keeps every not-yet-measured item explicit', () => {
    expect(text).toContain('Uncertainty: not yet measured');
    for (const k of Object.keys(real.not_yet_measured)) expect(text).toContain(`${k.replace(/_/g, ' ')}: not yet measured`);
    expect(text).toMatch(/Real-world validation: not yet measured/);
  });
});

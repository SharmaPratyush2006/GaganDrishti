/**
 * Step 6: Phase 5 error / confidence overlays.
 * Deterministic in-memory rasters + the tiny_confidence.tif fixture; the real
 * Phase 5/6 outputs are used at the end when present (skipped otherwise).
 */
import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { heightmapFromRaster } from '../src/3d/heightmap.js';
import {
  absPercentile, buildOverlay, compositeTexels, CONFIDENCE_DISCLAIMER, confidenceOverlayData, confidenceStateTable,
  diagnosticEligibility, divergingColor, ERROR_LABEL, errorOverlayData, overlayValueAt, STATE_COLORS,
} from '../src/3d/overlays.js';
import { initialCameraPose } from '../src/3d/scene.js';
import { TerrainLOD } from '../src/3d/terrain.js';
import { createDataTexture, NODATA_TEXEL } from '../src/3d/texture.js';
import { readGeoTiff } from '../src/data/geotiffLoader.js';
import { loadDemo } from '../src/data/load.js';
import { makeRaster, validityMask } from '../src/data/raster.js';
import { diskFetch, expected, FIXTURES, hasPhase6Outputs, readArrayBuffer, REPO_ROOT } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

const T = expected['tiny_ndsm_cog.tif'].transform;
const GEO = { transform: T, crs: 'EPSG:32643', units: 'metres' };
// The Phase 5 state table, in the report's shape (validation/confidence.py documents the same codes).
const STATES = {
  INVALID: { code: 0, meaning: 'no valid prediction/reference at this pixel' },
  UNSUITABLE: { code: 1, meaning: 'water surface; a height here is not meaningful' },
  REDUCED: { code: 2, meaning: 'shadow-occluded and/or scene sun elevation outside the usable band' },
  NOT_ASSESSED: { code: 3, meaning: "no condition fired, but at least one check's input was unavailable" },
  HIGH: { code: 4, meaning: 'every check ran and none fired' },
};
const STATE_TAGS = { STATE_0: 'INVALID', STATE_1: 'UNSUITABLE', STATE_2: 'REDUCED', STATE_3: 'NOT_ASSESSED', STATE_4: 'HIGH' };
const SRC = 'data/outputs/phase4b/synthetic_dsm.tif';
const products = { sourceDsm: SRC, confidenceStates: STATES };

function dsm(rows, opts = {}) {
  const r = rasterFrom(rows, { ...GEO, productType: 'dsm', ...opts });
  r.tags = { SOURCE_DSM: 'data\\outputs\\phase4b\\synthetic_dsm.tif' };
  return r;
}
function errorRaster(rows, opts = {}) {
  const r = rasterFrom(rows, { ...GEO, productType: 'error', ...opts });
  r.tags = { CONTENT: 'error = prediction - reference (positive = over-prediction)', PRODUCT: 'dsm', UNITS: 'metres' };
  return r;
}
function confidenceRaster(rows, { transform = T, crs = 'EPSG:32643', tags = STATE_TAGS, nodata = null } = {}) {
  const height = rows.length;
  const width = rows[0].length;
  const values = Uint8Array.from(rows.flat());
  return makeRaster({ width, height, values, valid: validityMask(values, nodata), dtype: 'uint8', nodata, crs, transform,
    productType: 'confidence', tags: { CONTENT: 'Phase 5 confidence proxy (rule-based flags, NOT a probability)', ...tags }, source: 'test' });
}
const flat = (h, w, v) => grid(h, w, () => v);

describe('eligibility: only an aligned Phase 5 diagnostic on the DSM it validated', () => {
  const surface = dsm(flat(6, 8, 500));
  const err = errorRaster(flat(6, 8, 0.1));

  it('compatible aligned error raster on the validated DSM is available', () => {
    const e = diagnosticEligibility('error', { surfaceRaster: surface, diagnostic: err, products });
    expect(e.available).toBe(true);
    expect(e.alignment.aligned).toBe(true);
  });

  it('DSM error is never presented as nDSM (or DTM) error', () => {
    const ndsm = rasterFrom(flat(6, 8, 3), GEO);
    const e = diagnosticEligibility('error', { surfaceRaster: ndsm, diagnostic: err, products });
    expect(e.available).toBe(false);
    expect(e.reason).toBe('Error overlay unavailable for this nDSM — Phase 5 error map validates the source DSM, not nDSM. Switch the surface to DSM to view it.');
    const dtm = rasterFrom(flat(6, 8, 3), { ...GEO, productType: 'dtm' });
    expect(diagnosticEligibility('error', { surfaceRaster: dtm, diagnostic: err, products }).reason).toMatch(/for this DTM — .*not DTM/);
  });

  it('Phase 5 source-DSM confidence is not presented as nDSM confidence', () => {
    const ndsm = rasterFrom(flat(6, 8, 3), GEO);
    const e = diagnosticEligibility('confidence', { surfaceRaster: ndsm, diagnostic: confidenceRaster(flat(6, 8, 4)), products });
    expect(e.available).toBe(false);
    expect(e.reason).toMatch(/^Confidence overlay unavailable for this nDSM — available only for the Phase 5 source DSM; not an nDSM confidence map\./);
  });

  it('no linked raster -> unavailable, with the loader\'s reason', () => {
    const e = diagnosticEligibility('error', { surfaceRaster: surface, diagnostic: null, notAvailable: { error: 'no Phase 5 report validated X' } });
    expect(e.available).toBe(false);
    expect(e.reason).toBe('Error overlay unavailable — no aligned error raster: no Phase 5 report validated X');
  });

  it('misaligned rasters are rejected: shifted transform, other size, other CRS', () => {
    const shifted = errorRaster(flat(6, 8, 0.1), { transform: [0.5, 0, 700000.5, 0, -0.5, 3170000] });
    const bigger = errorRaster(flat(7, 8, 0.1));
    const utm44 = errorRaster(flat(6, 8, 0.1), { crs: 'EPSG:32644' });
    for (const [bad, why] of [[shifted, /transform/], [bigger, /size/], [utm44, /CRS/]]) {
      const e = diagnosticEligibility('error', { surfaceRaster: surface, diagnostic: bad, products });
      expect(e.available).toBe(false);
      expect(e.reason).toMatch(/no aligned error raster/);
      expect(e.reason).toMatch(why);
    }
    const c = diagnosticEligibility('confidence', { surfaceRaster: surface, diagnostic: confidenceRaster(flat(6, 8, 4), { transform: [0.5, 0, 700001, 0, -0.5, 3170000] }), products });
    expect(c.available).toBe(false);
  });

  it('a DSM exported from a different source DSM, or an error map of another product, is rejected', () => {
    const other = dsm(flat(6, 8, 500));
    other.tags = { SOURCE_DSM: 'data/outputs/phase4b/synthetic_dsm_known_calibration.tif' };
    expect(diagnosticEligibility('error', { surfaceRaster: other, diagnostic: err, products }).reason).toMatch(/exported from .*known_calibration.*Phase 5 validated/);
    const agl = errorRaster(flat(6, 8, 0.1));
    agl.tags.PRODUCT = 'agl';
    expect(diagnosticEligibility('error', { surfaceRaster: surface, diagnostic: agl, products }).reason).toMatch(/validates a agl product, not a DSM/);
  });

  it('a Phase 3 relative raster gets no Phase 5 overlay', () => {
    const rel = rasterFrom(flat(4, 4, 0.3), { productType: 'relative_height' });
    const e = diagnosticEligibility('error', { surfaceRaster: rel, diagnostic: null, notAvailable: { error: 'a Phase 3 relative raster has no Phase 5 error map' } });
    expect(e.available).toBe(false);
    expect(e.reason).toMatch(/Phase 3 relative raster/);
  });

  it('the fixture tiny_confidence.tif (8x8) is misaligned with the 20x12 nDSM fixture grid', async () => {
    const conf = await readGeoTiff(readArrayBuffer(path.join(FIXTURES, 'tiny_confidence.tif')), { productType: 'confidence' });
    const s = dsm(flat(12, 20, 500));
    const e = diagnosticEligibility('confidence', { surfaceRaster: s, diagnostic: conf, products });
    expect(e.available).toBe(false);
    expect(e.reason).toMatch(/size 8×8 ≠ raster 20×12/);
  });
});

describe('error overlay colours', () => {
  it('signed error is preserved: positive red, negative blue, zero neutral', () => {
    const err = errorRaster([[0.2, -0.2, 0, 0.05, -0.05, 1.0]]);
    const ov = errorOverlayData(err);
    const px = (i) => Array.from(ov.data.slice(i * 4, i * 4 + 4));
    const [pos, neg, zero, smallPos, smallNeg] = [0, 1, 2, 3, 4].map(px);
    expect(pos[0]).toBeGreaterThan(pos[2]);
    expect(smallPos[0]).toBeGreaterThan(smallPos[2]);
    expect(neg[2]).toBeGreaterThan(neg[0]);
    expect(smallNeg[2]).toBeGreaterThan(smallNeg[0]);
    expect(zero[0]).toBe(zero[2]);
    expect(zero[0]).toBe(zero[1]);
    expect([pos, neg, zero].every((p) => p[3] === 255)).toBe(true);
    expect(ov.counts).toMatchObject({ positive: 3, negative: 2, zero: 1, invalid: 0 });
    expect([ov.min, ov.max]).toEqual([Math.fround(-0.2), 1]);
  });

  it('diverging ramp is antisymmetric in hue and saturates at the limit', () => {
    for (const v of [0.01, 0.3, 0.9]) {
      const [r1, , b1] = divergingColor(v, 1);
      const [r2, , b2] = divergingColor(-v, 1);
      expect(r1 > b1 && b2 > r2).toBe(true);
    }
    expect(divergingColor(5, 1)).toEqual(divergingColor(1, 1));
  });

  it('saturation limit is the 99th percentile of |error| over valid pixels only', () => {
    const vals = grid(11, 10, (r, c) => (r === 10 ? null : r * 10 + c + 1)); // 1..100, plus a row of nodata (ignored)
    const err = errorRaster(vals);
    expect(absPercentile(err, 99)).toBe(99);
    expect(errorOverlayData(err).limit).toBe(99);
  });

  it('nodata and non-finite pixels are not drawn and nothing replaces them', () => {
    const err = errorRaster([[null, 0.1, -0.1]]);
    err.values[2] = NaN;
    err.valid[2] = 1; // defensive: bypass the loader's mask
    const ov = errorOverlayData(err);
    expect(ov.data[3]).toBe(0);
    expect(ov.data[11]).toBe(0);
    expect(Array.from(ov.data.slice(0, 3))).toEqual([0, 0, 0]);
    expect(ov.counts.invalid).toBe(2);
    // Blended: alpha 0 leaves the base texel exactly as it was.
    const base = Uint8Array.from([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255]);
    const out = compositeTexels(base, ov.data, 1);
    expect(Array.from(out.slice(0, 4))).toEqual([10, 20, 30, 255]);
    expect(Array.from(out.slice(8, 12))).toEqual([70, 80, 90, 255]);
    expect(Array.from(out.slice(4, 7))).toEqual(Array.from(ov.data.slice(4, 7)));
    // Without a base texture, the untextured surface colour shows through.
    expect(Array.from(compositeTexels(null, ov.data, 1).slice(0, 3))).toEqual(NODATA_TEXEL.slice(0, 3));
  });

  it('overlay values are read raw, signed, with units', () => {
    const surface = dsm([[500, 501, 502]]);
    const err = errorRaster([[0.25, -0.125, null]]);
    const ov = buildOverlay('error', { surfaceRaster: surface, diagnostic: err, products });
    expect(ov.available).toBe(true);
    expect(ov.label).toBe(ERROR_LABEL);
    expect(overlayValueAt(ov, err, 0, 0)).toEqual({ text: '+0.250 m', value: 0.25 });
    expect(overlayValueAt(ov, err, 0, 1)).toEqual({ text: '-0.125 m', value: -0.125 });
    expect(overlayValueAt(ov, err, 0, 2)).toEqual({ text: 'nodata', value: null });
  });

  it('building an overlay does not modify the source raster', () => {
    const err = errorRaster(grid(8, 8, (r, c) => (r - c) / 10));
    const before = Float32Array.from(err.values);
    buildOverlay('error', { surfaceRaster: dsm(flat(8, 8, 500)), diagnostic: err, products });
    expect(err.values).toEqual(before);
  });
});

describe('overlay on the terrain: appearance only', () => {
  function scene() {
    const surface = dsm(grid(65, 65, (r, c) => 500 + ((r * 3 + c) % 7)));
    const hm = heightmapFromRaster(surface);
    const terrain = new TerrainLOD(hm, { tileSize: 16 });
    terrain.update(initialCameraPose(terrain.bounds()).position);
    return { surface, terrain };
  }

  it('does not modify terrain heights, rebuild tiles or change the LOD selection', () => {
    const { surface, terrain } = scene();
    const err = errorRaster(grid(65, 65, (r, c) => (r - c) / 50));
    const positions = terrain.group.children.map((m) => Float32Array.from(m.geometry.attributes.position.array));
    const geometries = terrain.group.children.map((m) => m.geometry);
    const built = terrain.counters.tilesBuiltTotal;
    const cacheSize = terrain.cache.size;
    const valuesBefore = Float32Array.from(surface.values);

    const ov = buildOverlay('error', { surfaceRaster: surface, diagnostic: err, products });
    // What Viewer3D does on toggle: a new texture on the material, nothing else.
    const tex = createDataTexture({ width: 65, height: 65, data: compositeTexels(null, ov.overlay.data, 0.65) });
    terrain.setTexture(tex);
    expect(terrain.group.children.every((m) => m.material.map === tex)).toBe(true);
    terrain.setTexture(null);
    tex.dispose();
    terrain.update(initialCameraPose(terrain.bounds()).position);

    expect(terrain.counters.tilesBuiltTotal).toBe(built);
    expect(terrain.cache.size).toBe(cacheSize);
    expect(terrain.group.children.map((m) => m.geometry)).toEqual(geometries);
    terrain.group.children.forEach((m, i) => expect(m.geometry.attributes.position.array).toEqual(positions[i]));
    expect(surface.values).toEqual(valuesBefore);
    terrain.dispose();
  });

  it('the overlay texel of pixel (row, col) is the texel every tile samples for that pixel', () => {
    const { terrain } = scene();
    const W = 65;
    for (const mesh of terrain.group.children) {
      const uv = mesh.geometry.attributes.uv.array;
      const pix = mesh.geometry.attributes.pixelIndex.array;
      for (let v = 0; v < pix.length; v += 37) {
        const row = Math.floor(pix[v] / W);
        const col = pix[v] % W;
        // texture.js / overlays.js layout: texel (row, col) centre at ((col+.5)/W, (row+.5)/H), flipY false.
        expect(Math.floor(uv[v * 2] * W)).toBe(col);
        expect(Math.floor(uv[v * 2 + 1] * 65)).toBe(row);
      }
    }
    terrain.dispose();
  });
});

describe('confidence overlay: documented rule-based states', () => {
  it('state table comes from the Phase 5 report and matches the file tags', () => {
    const t = confidenceStateTable(STATES, STATE_TAGS);
    expect(t.states.map((s) => [s.code, s.name, s.meaning])).toEqual(
      Object.entries(STATES).map(([name, s]) => [s.code, name, s.meaning]));
    expect(t.states.map((s) => s.color)).toEqual(Object.keys(STATES).map((n) => STATE_COLORS[n]));
  });

  it('a report/tag disagreement is refused, not guessed', () => {
    const t = confidenceStateTable(STATES, { ...STATE_TAGS, STATE_2: 'HIGH' });
    expect(t.error).toMatch(/STATE_2=HIGH but the Phase 5 report says REDUCED/);
    expect(confidenceStateTable(null, {}).error).toMatch(/no documented state table/);
  });

  it('each code maps to its own state colour; undocumented codes and nodata are not drawn', () => {
    const conf = confidenceRaster([[0, 1, 2, 3, 4, 9]]);
    const { states } = confidenceStateTable(STATES, STATE_TAGS);
    const ov = confidenceOverlayData(conf, states);
    for (const s of states) {
      expect(Array.from(ov.data.slice(s.code * 4, s.code * 4 + 4))).toEqual([...STATE_COLORS[s.name], 255]);
      expect(ov.counts[s.name]).toBe(1);
    }
    expect(ov.data[5 * 4 + 3]).toBe(0);
    expect(ov.undocumented).toBe(1);
    const withNodata = confidenceRaster([[4, 255]], { nodata: 255 });
    const ov2 = confidenceOverlayData(withNodata, states);
    expect(ov2.nodata).toBe(1);
    expect(ov2.data[7]).toBe(0);
  });

  it('legend matches the documented states, carries the disclaimer, and produces no probabilities', () => {
    const surface = dsm(flat(2, 3, 500));
    const conf = confidenceRaster([[0, 2, 3], [4, 4, 1]]);
    const ov = buildOverlay('confidence', { surfaceRaster: surface, diagnostic: conf, products, confidenceStates: STATES });
    expect(ov.available).toBe(true);
    expect(ov.disclaimer).toBe('Rule-based quality flags, not probabilities.');
    expect(CONFIDENCE_DISCLAIMER).toBe(ov.disclaimer);
    expect(ov.states.map((s) => s.name)).toEqual(['INVALID', 'UNSUITABLE', 'REDUCED', 'NOT_ASSESSED', 'HIGH']);
    const { overlay, ...rest } = ov;
    const text = JSON.stringify({ ...rest, counts: overlay.counts });
    expect(text).not.toMatch(/%|percent|probability(?! )|likelihood|sigma|±/i);
    expect(Object.values(overlay.counts).every(Number.isInteger)).toBe(true);
    expect(overlayValueAt(ov, conf, 0, 1)).toEqual({ text: 'REDUCED (code 2)', value: 2, state: 'REDUCED' });
  });

  it('the confidence fixture decodes to its documented states', async () => {
    const conf = await readGeoTiff(readArrayBuffer(path.join(FIXTURES, 'tiny_confidence.tif')), { productType: 'confidence' });
    const surface = dsm(flat(8, 8, 500));
    const ov = buildOverlay('confidence', { surfaceRaster: surface, diagnostic: conf, products, confidenceStates: STATES });
    expect(ov.available).toBe(true);
    const want = expected['tiny_confidence.tif'].values.flat();
    const names = ['INVALID', 'UNSUITABLE', 'REDUCED', 'NOT_ASSESSED', 'HIGH'];
    for (const n of names) expect(ov.overlay.counts[n]).toBe(want.filter((v) => names[v] === n).length);
  });
});

describe.skipIf(!hasPhase6Outputs())('real Phase 5 / Phase 6 outputs', () => {
  const fetchImpl = diskFetch();
  const ctxFor = (d, surface) => ({
    surfaceRaster: d.rasters[surface], products: d.products, notAvailable: d.notAvailable, confidenceStates: d.products.confidenceStates,
  });

  it('acceptance product: nDSM shown -> both overlays refused with the DSM-vs-nDSM reason', async () => {
    const d = await loadDemo({}, fetchImpl);
    for (const kind of ['error', 'confidence']) {
      const ov = buildOverlay(kind, { ...ctxFor(d, 'ndsm'), diagnostic: d.rasters[kind] });
      expect(ov.available).toBe(false);
      expect(ov.reason).toMatch(/for this nDSM/);
      const onDsm = buildOverlay(kind, { ...ctxFor(d, 'dsm'), diagnostic: d.rasters[kind] });
      if (!d.rasters[kind]) expect(onDsm.reason).toMatch(/no aligned/);
    }
  });

  it('a product whose source DSM Phase 5 validated: overlays available on its DSM only, legend = report states', async () => {
    const report = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'data/outputs/phase6/synthetic/phase6_report.json'), 'utf-8'));
    const names = Object.keys(report.products);
    let found = 0;
    for (const name of names) {
      const d = await loadDemo({ productName: name }, fetchImpl);
      if (!d.rasters.error) continue;
      found++;
      const err = buildOverlay('error', { ...ctxFor(d, 'dsm'), diagnostic: d.rasters.error });
      expect(err.available).toBe(true);
      // Sign preserved on the real map: every drawn pixel's hue follows its value's sign.
      const e = d.rasters.error;
      for (let i = 0; i < e.values.length; i += 97) {
        const [r, , b] = err.overlay.data.slice(i * 4, i * 4 + 3);
        if (e.values[i] > 0) expect(r).toBeGreaterThanOrEqual(b);
        if (e.values[i] < 0) expect(b).toBeGreaterThanOrEqual(r);
      }
      const conf = buildOverlay('confidence', { ...ctxFor(d, 'dsm'), diagnostic: d.rasters.confidence });
      expect(conf.available).toBe(true);
      const r5 = d.products.confidenceStates;
      expect(conf.states.map((s) => [s.name, s.code, s.meaning])).toEqual(
        Object.entries(r5).map(([n, s]) => [n, s.code, s.meaning]).sort((a, b) => a[1] - b[1]));
      // Same product, nDSM displayed: refused even though the DSM diagnostics exist.
      expect(buildOverlay('error', { ...ctxFor(d, 'ndsm'), diagnostic: d.rasters.error }).reason)
        .toBe('Error overlay unavailable for this nDSM — Phase 5 error map validates the source DSM, not nDSM. Switch the surface to DSM to view it.');
    }
    expect(found).toBeGreaterThan(0);
  });
});

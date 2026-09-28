import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { readGeoTiff } from '../src/data/geotiffLoader.js';
import { validStats, valueAt } from '../src/data/raster.js';
import { expected, FIXTURES, readArrayBuffer } from './helpers.js';

const load = (name, opts = {}) => readGeoTiff(readArrayBuffer(path.join(FIXTURES, name)), { source: name, ...opts });

/** Every pixel equals GDAL's reading of the same file (null = nodata). */
function expectPixelsMatch(raster, exp) {
  for (let r = 0; r < exp.height; r++) {
    for (let c = 0; c < exp.width; c++) {
      const want = exp.values[r][c];
      const got = valueAt(raster, r, c);
      if (want === null) expect(got, `(${r},${c})`).toBeNull();
      else expect(got, `(${r},${c})`).toBe(want);
    }
  }
}

describe('readGeoTiff on a pipeline-written COG', () => {
  const exp = expected['tiny_ndsm_cog.tif'];

  it('preserves raster dimensions', async () => {
    const r = await load('tiny_ndsm_cog.tif');
    expect([r.width, r.height]).toEqual([exp.width, exp.height]);
    expect(r.values.length).toBe(exp.width * exp.height);
  });

  it('preserves nodata explicitly and matches GDAL pixel for pixel', async () => {
    const r = await load('tiny_ndsm_cog.tif');
    expect(r.nodata).toBe(exp.nodata);
    expectPixelsMatch(r, exp);
    expect(validStats(r).nodataPixels).toBe(3);
    // A real zero is valid; a nodata pixel is null, not 0.
    expect(valueAt(r, 0, 0)).toBe(0);
    expect(valueAt(r, 5, 7)).toBeNull();
  });

  it('carries CRS, transform, units, product type and mode', async () => {
    const r = await load('tiny_ndsm_cog.tif');
    expect(r.crs).toBe(exp.crs);
    expect(r.transform).toEqual(exp.transform);
    expect(r.units).toBe('metres');
    expect(r.productType).toBe('ndsm');
    expect(r.mode).toBe('ABSOLUTE');
    expect(r.synthetic).toBe(true);
    expect(r.dtype).toBe('float32');
  });

  it('rejects a caller expectation that contradicts the file', async () => {
    await expect(load('tiny_ndsm_cog.tif', { productType: 'dsm' })).rejects.toThrow(/declares ndsm/);
  });
});

describe('readGeoTiff edge cases', () => {
  it('treats a pixel-space grid with no CRS as RELATIVE even if it says metres', async () => {
    const r = await load('tiny_nocrs.tif');
    expect(r.crs).toBeNull();
    expect(r.transform).toBeNull();
    expect(r.mode).toBe('RELATIVE');
    expect(r.modeReason).toMatch(/no CRS/);
    expectPixelsMatch(r, expected['tiny_nocrs.tif']);
  });

  it('keeps a uint8 raster with no declared nodata fully valid (codes interpreted later)', async () => {
    const r = await load('tiny_confidence.tif');
    expect(r.nodata).toBeNull();
    expect(r.productType).toBe('confidence');
    expect(r.mode).toBeNull();
    expect(r.tags.STATE_0).toBe('INVALID');
    expectPixelsMatch(r, expected['tiny_confidence.tif']);
  });

  it('marks sparse (unwritten) blocks invalid instead of reading them as 0', async () => {
    const exp = expected['tiny_sparse.tif'];
    expect(exp.block_byte_counts_row_major.filter((n) => n === 0).length).toBe(3);
    const r = await load('tiny_sparse.tif');
    expect(valueAt(r, 0, 0)).toBe(7);
    expect(valueAt(r, 15, 15)).toBe(7);
    expect(valueAt(r, 0, 16)).toBeNull();
    expect(valueAt(r, 16, 0)).toBeNull();
    expect(valueAt(r, 31, 31)).toBeNull();
    expect(validStats(r).validPixels).toBe(16 * 16);
    expect(r.warnings.join()).toMatch(/3 sparse/);
  });

  it('fails clearly on bytes that are not a TIFF', async () => {
    await expect(readGeoTiff(new ArrayBuffer(16), { source: 'junk' })).rejects.toThrow(/junk: not a readable TIFF/);
  });
});

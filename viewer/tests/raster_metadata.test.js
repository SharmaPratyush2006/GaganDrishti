import { describe, expect, it } from 'vitest';

import {
  affineFromTiffTags,
  crsFromGeoKeys,
  determineMode,
  formatValue,
  inferProductType,
  reconcileProductType,
  unitSuffix,
} from '../src/data/metadata.js';
import {
  makeRaster,
  mapToPixel,
  mapToPixelIndex,
  pixelToMap,
  sameGrid,
  validityMask,
  valueAt,
} from '../src/data/raster.js';

const T = [0.5, 0, 700000, 0, -0.5, 3170000];

describe('validityMask', () => {
  it('invalidates declared nodata and non-finite values, keeps real zeros', () => {
    const v = new Float32Array([0, -9999, NaN, Infinity, 1.5]);
    expect(Array.from(validityMask(v, -9999))).toEqual([1, 0, 0, 0, 1]);
  });
  it('with no nodata declared, only non-finite values are invalid', () => {
    expect(Array.from(validityMask(new Float32Array([0, -9999, NaN]), null))).toEqual([1, 1, 0]);
  });
  it('compares float32 values against the float32-rounded nodata', () => {
    expect(Array.from(validityMask(new Float32Array([0.1, 0.2]), 0.1))).toEqual([0, 1]);
  });
});

describe('pixel <-> map coordinates', () => {
  it('pixel centre maps to the expected easting/northing and back', () => {
    const [x, y] = pixelToMap(T, 100.5, 100.5);
    expect([x, y]).toEqual([700050.25, 3169949.75]);
    expect(mapToPixel(T, x, y)).toEqual([100.5, 100.5]);
  });
  it('integer lookup floors like rasterio.rowcol and returns null outside', () => {
    const r = makeRaster({ width: 512, height: 512, values: new Float32Array(512 * 512), valid: new Uint8Array(512 * 512), transform: T });
    expect(mapToPixelIndex(r, 700050.0, 3169950.0)).toEqual({ row: 100, col: 100 });
    expect(mapToPixelIndex(r, 699999.9, 3169950.0)).toBeNull();
    expect(mapToPixelIndex(r, 700256.0, 3169950.0)).toBeNull();
  });
  it('inverts a rotated transform', () => {
    const rot = [0.3, 0.4, 10, 0.4, -0.3, 20];
    const [x, y] = pixelToMap(rot, 7.25, 3.5);
    const [c, r] = mapToPixel(rot, x, y);
    expect(c).toBeCloseTo(7.25, 12);
    expect(r).toBeCloseTo(3.5, 12);
  });
});

describe('valueAt', () => {
  const r = makeRaster({ width: 2, height: 2, values: new Float32Array([0, -9999, 3, 4]), valid: validityMask(new Float32Array([0, -9999, 3, 4]), -9999) });
  it('returns null for nodata, never 0 or the nodata value', () => {
    expect(valueAt(r, 0, 1)).toBeNull();
  });
  it('returns a real zero as 0', () => {
    expect(valueAt(r, 0, 0)).toBe(0);
  });
  it('returns null outside the raster and for non-integer indices', () => {
    expect(valueAt(r, 2, 0)).toBeNull();
    expect(valueAt(r, -1, 0)).toBeNull();
    expect(valueAt(r, 0.5, 0)).toBeNull();
  });
});

describe('makeRaster', () => {
  it('rejects values whose length does not match the size', () => {
    expect(() => makeRaster({ width: 3, height: 2, values: new Float32Array(5), valid: new Uint8Array(5) })).toThrow(/expected 3 x 2/);
  });
});

describe('sameGrid', () => {
  const base = { width: 4, height: 4, values: new Float32Array(16), valid: new Uint8Array(16), crs: 'EPSG:32643', transform: T };
  it('is true for identical grids and false when any of size/CRS/transform differ', () => {
    expect(sameGrid(makeRaster(base), makeRaster(base))).toBe(true);
    expect(sameGrid(makeRaster(base), makeRaster({ ...base, crs: 'EPSG:4326' }))).toBe(false);
    expect(sameGrid(makeRaster(base), makeRaster({ ...base, transform: [0.5, 0, 700000.5, 0, -0.5, 3170000] }))).toBe(false);
    expect(sameGrid(makeRaster(base), makeRaster({ ...base, width: 2, height: 8 }))).toBe(false);
  });
});

describe('metadata', () => {
  it('reads EPSG codes from GeoKeys and reports absence as null', () => {
    expect(crsFromGeoKeys({ ProjectedCSTypeGeoKey: 32643 })).toBe('EPSG:32643');
    expect(crsFromGeoKeys({ GeographicTypeGeoKey: 4326 })).toBe('EPSG:4326');
    expect(crsFromGeoKeys(null)).toBeNull();
  });
  it('builds the GDAL-order affine from tiepoint + scale, and from ModelTransformation', () => {
    expect(affineFromTiffTags({ pixelScale: [0.5, 0.5, 0], tiepoint: [0, 0, 0, 700000, 3170000, 0] })).toEqual(T);
    const m = [0.3, 0.4, 0, 10, 0.4, -0.3, 0, 20, 0, 0, 0, 0, 0, 0, 0, 1];
    expect(affineFromTiffTags({ modelTransformation: m })).toEqual([0.3, 0.4, 10, 0.4, -0.3, 20]);
    expect(affineFromTiffTags({})).toBeNull();
  });
  it('shifts PixelIsPoint by half a pixel, as GDAL does', () => {
    expect(affineFromTiffTags({ pixelScale: [1, 1, 0], tiepoint: [0, 0, 0, 100, 200, 0], pixelIsPoint: true }))
      .toEqual([1, 0, 99.5, 0, -1, 200.5]);
  });
  it('infers the product from the pipeline CONTENT tag and keeps DSM, DTM, nDSM and AGL distinct', () => {
    expect(inferProductType({ CONTENT: 'DSM: absolute surface elevation (input, re-exported)' })).toBe('dsm');
    expect(inferProductType({ CONTENT: 'DSM = T + a*exp(z_rel) + b' })).toBe('dsm');
    expect(inferProductType({ CONTENT: 'DTM: absolute bare-ground elevation' })).toBe('dtm');
    expect(inferProductType({ CONTENT: 'nDSM = DSM - DTM: object height above the extracted ground' })).toBe('ndsm');
    expect(inferProductType({ CONTENT: 'AGL = a*exp(z_rel) + b (Phase 4a synthetic calibration)' })).toBe('agl');
    expect(inferProductType({ CONTENT: 'error = prediction - reference (positive = over-prediction)' })).toBe('error');
    expect(inferProductType({ STATE_0: 'INVALID' })).toBe('confidence');
    expect(inferProductType({})).toBe('unknown');
  });
  it('refuses a product whose file declares a different type than expected', () => {
    expect(() => reconcileProductType('ndsm', 'dsm', 'x.tif')).toThrow(/expected a ndsm product but the file declares dsm/);
    expect(reconcileProductType('image', 'unknown', 'x.tif')).toBe('image');
  });
});

describe('ABSOLUTE / RELATIVE', () => {
  it('is ABSOLUTE only with CRS, transform and metres', () => {
    expect(determineMode({ crs: 'EPSG:32643', transform: T, units: 'metres', productType: 'ndsm' }).mode).toBe('ABSOLUTE');
    expect(determineMode({ crs: null, transform: T, units: 'metres', productType: 'ndsm' }).mode).toBe('RELATIVE');
    expect(determineMode({ crs: 'EPSG:32643', transform: null, units: 'metres', productType: 'ndsm' }).mode).toBe('RELATIVE');
    expect(determineMode({ crs: 'EPSG:32643', transform: T, units: null, productType: 'ndsm' }).mode).toBe('RELATIVE');
  });
  it('does not assign a height mode to images or confidence flags', () => {
    expect(determineMode({ crs: 'EPSG:32643', transform: T, units: null, productType: 'image' }).mode).toBeNull();
    expect(determineMode({ crs: 'EPSG:32643', transform: T, units: null, productType: 'confidence' }).mode).toBeNull();
  });
  it('never formats a RELATIVE value with metres', () => {
    const rel = { mode: 'RELATIVE', units: 'metres' }; // even if a stray tag claims metres
    expect(unitSuffix(rel)).toBe('');
    const s = formatValue(1.23456, rel);
    expect(s).toContain('unitless');
    expect(s).not.toMatch(/\bm\b|metre|meter/);
  });
  it('formats ABSOLUTE values in metres and nodata as "nodata"', () => {
    const abs = { mode: 'ABSOLUTE', units: 'metres' };
    expect(formatValue(12.434, abs)).toBe('12.43 m');
    expect(formatValue(null, abs)).toBe('nodata');
  });
});

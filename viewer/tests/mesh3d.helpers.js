import { determineMode } from '../src/data/metadata.js';
import { makeRaster, validityMask } from '../src/data/raster.js';

/**
 * A small raster record built exactly as Step 2 builds one.
 * @param {(number|null)[][]} rows  null = nodata
 */
export function rasterFrom(rows, { transform = null, crs = null, units = null, productType = 'ndsm', nodata = -9999 } = {}) {
  const height = rows.length;
  const width = rows[0].length;
  const values = new Float32Array(width * height);
  rows.flat().forEach((v, i) => { values[i] = v === null ? nodata : v; });
  const { mode, reason } = determineMode({ crs, transform, units, productType });
  return makeRaster({
    width, height, values, valid: validityMask(values, nodata), dtype: 'float32', nodata, crs, transform,
    units, productType, mode, modeReason: reason, source: 'test',
  });
}

export const grid = (h, w, f) => Array.from({ length: h }, (_, r) => Array.from({ length: w }, (_, c) => f(r, c)));

/** Triangles of a tile mesh as [[r,c],[r,c],[r,c]] sample coordinates. */
export function trianglesRC(tile, width) {
  const out = [];
  for (let t = 0; t < tile.indices.length; t += 3) {
    out.push([0, 1, 2].map((k) => {
      const p = tile.pixelIndex[tile.indices[t + k]];
      return [Math.floor(p / width), p % width];
    }));
  }
  return out;
}

/** Signed area in (col, row) index space. */
export function areaRC([[r0, c0], [r1, c1], [r2, c2]]) {
  return ((c1 - c0) * (r2 - r0) - (c2 - c0) * (r1 - r0)) / 2;
}

/** Point (r, c) strictly inside or on a triangle (index space). */
export function covers(tri, r, c) {
  const [[r0, c0], [r1, c1], [r2, c2]] = tri;
  const s = (ar, ac, br, bc) => (bc - ac) * (r - ar) - (c - ac) * (br - ar);
  const d1 = s(r0, c0, r1, c1);
  const d2 = s(r1, c1, r2, c2);
  const d3 = s(r2, c2, r0, c0);
  const neg = d1 < 0 || d2 < 0 || d3 < 0;
  const pos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(neg && pos);
}

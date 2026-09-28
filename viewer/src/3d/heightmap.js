/**
 * Raster record -> heightmap: where each sample sits in the 3D world.
 *
 * In plain words: every raster pixel becomes one 3D point. Its horizontal
 * position comes from the raster's own transform (pixel centre -> map
 * coordinate); its height is the raster value itself. Nothing is reloaded:
 * this wraps the Step 2 raster record.
 *
 * World axes (Three.js is Y-up):
 *   x =  (map x - origin x)       east
 *   z = -(map y - origin y)       south (so north is -z)
 *   y =  value * verticalScale
 *
 * `origin` is the map coordinate of the raster centre. It is subtracted only
 * so float32 positions keep sub-millimetre precision (a UTM easting of 700000
 * would otherwise be rounded to ~6 cm); map = world + origin recovers it.
 *
 * Vertical scale:
 *   ABSOLUTE -> 1 by default (1:1 metres). Values are stored unchanged.
 *   RELATIVE -> values are unitless and the horizontal axes are pixels, so
 *               there is no common unit; a display-only exaggeration is chosen
 *               (relief = 10% of the horizontal extent) and reported. The
 *               raw value is value = y / verticalScale (exact at scale 1;
 *               within float32 rounding of the scaled value otherwise).
 *
 * A raster with no transform (e.g. a Phase 3 .npy) is laid out in pixel
 * space: x = column, z = row, labelled "pixels" -- never metres.
 */
import { HEIGHT_LIKE } from '../data/metadata.js';
import { pixelToMap, validStats } from '../data/raster.js';

/** Pixel space when there is no georeferencing: x = col, map y = -row. */
const PIXEL_SPACE = [1, 0, 0, 0, -1, 0];
const RELATIVE_RELIEF_FRACTION = 0.1;

export class HeightmapError extends Error {}

/**
 * @typedef {object} Heightmap
 * @property {import('../data/raster.js').Raster} raster
 * @property {number} width
 * @property {number} height
 * @property {number[]} transform        the transform actually used
 * @property {boolean} georeferenced
 * @property {string} horizontalUnits
 * @property {[number, number]} origin  map coordinate at world (0, 0)
 * @property {number} verticalScale
 * @property {string} verticalScaleReason
 * @property {boolean} flipWinding      true when the transform mirrors the grid
 * @property {{validPixels:number,nodataPixels:number,min:number|null,max:number|null}} stats
 * @property {Int32Array} invalidSat    (W+1)*(H+1) summed-area table of invalid pixels
 */

/**
 * @param {import('../data/raster.js').Raster} raster
 * @param {{verticalScale?: number}} [options]
 * @returns {Heightmap}
 */
export function heightmapFromRaster(raster, { verticalScale } = {}) {
  if (!HEIGHT_LIKE.has(raster.productType) || raster.mode === null) {
    throw new HeightmapError(`${raster.productType} is not a height product; it cannot be meshed as a surface`);
  }
  const { width, height } = raster;
  const georeferenced = raster.transform != null;
  const transform = georeferenced ? raster.transform : PIXEL_SPACE;
  const [a, b, , d, e] = transform;
  // Orientation of (col,row) -> (x,z) with z = -map y. Positive for north-up.
  const det = -(a * e - b * d);
  if (det === 0) throw new HeightmapError('transform is singular');

  const stats = validStats(raster);
  const origin = pixelToMap(transform, width / 2, height / 2);

  let scale;
  let reason;
  if (verticalScale !== undefined) {
    if (!(verticalScale > 0) || !Number.isFinite(verticalScale)) throw new HeightmapError(`invalid verticalScale ${verticalScale}`);
    scale = verticalScale;
    reason = 'set explicitly by the caller (display only; raw value = y / scale)';
  } else if (raster.mode === 'ABSOLUTE') {
    scale = 1;
    reason = '1:1 (metric values rendered unchanged)';
  } else {
    const extent = Math.max(width * Math.hypot(a, d), height * Math.hypot(b, e));
    const range = stats.max !== null ? stats.max - stats.min : 0;
    scale = range > 0 ? (RELATIVE_RELIEF_FRACTION * extent) / range : 1;
    reason = `display-only exaggeration: RELATIVE values are unitless and share no unit with the ${georeferenced ? 'map' : 'pixel'} axes; relief drawn as ${RELATIVE_RELIEF_FRACTION * 100}% of the horizontal extent`;
  }

  return {
    raster,
    width,
    height,
    transform,
    georeferenced,
    horizontalUnits: georeferenced ? `${raster.crs ?? 'unspecified CRS'} map units` : 'pixels (no georeferencing)',
    origin,
    verticalScale: scale,
    verticalScaleReason: reason,
    flipWinding: det < 0,
    stats,
    invalidSat: invalidSummedArea(raster),
  };
}

function invalidSummedArea({ width, height, valid }) {
  const W1 = width + 1;
  const sat = new Int32Array(W1 * (height + 1));
  for (let r = 0; r < height; r++) {
    let rowSum = 0;
    for (let c = 0; c < width; c++) {
      rowSum += valid[r * width + c] ? 0 : 1;
      sat[(r + 1) * W1 + c + 1] = sat[r * W1 + c + 1] + rowSum;
    }
  }
  return sat;
}

/**
 * True when every sample in rows r0..r1 and cols c0..c1 (inclusive) is valid.
 * @param {Heightmap} hm
 */
export function blockValid(hm, r0, c0, r1, c1) {
  const W1 = hm.width + 1;
  const s = hm.invalidSat;
  const n = s[(r1 + 1) * W1 + c1 + 1] - s[r0 * W1 + c1 + 1] - s[(r1 + 1) * W1 + c0] + s[r0 * W1 + c0];
  return n === 0;
}

/** @param {Heightmap} hm */
export function sampleValid(hm, row, col) {
  return hm.raster.valid[row * hm.width + col] === 1;
}

/**
 * World position of the sample at (row, col) -- the pixel centre.
 * Only meaningful for a valid sample; callers never place invalid ones.
 * @param {Heightmap} hm
 * @returns {[number, number, number]}
 */
export function samplePosition(hm, row, col) {
  const [x, z] = sampleXZ(hm, row, col);
  return [x, Number(hm.raster.values[row * hm.width + col]) * hm.verticalScale, z];
}

/**
 * Horizontal world position (x, z) of a sample; defined for any sample,
 * valid or not, because it depends only on the transform.
 * @returns {[number, number]}
 */
export function sampleXZ(hm, row, col) {
  const [mx, my] = pixelToMap(hm.transform, col + 0.5, row + 0.5);
  return [mx - hm.origin[0], -(my - hm.origin[1])];
}

/** World -> map coordinate (inverse of the horizontal part of samplePosition). */
export function worldToMap(hm, x, z) {
  return [x + hm.origin[0], -z + hm.origin[1]];
}

/** World y -> raster value (inverse of the vertical scale). */
export function worldYToValue(hm, y) {
  return y / hm.verticalScale;
}

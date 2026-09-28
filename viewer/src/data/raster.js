/**
 * The viewer's one in-memory raster representation, and pixel/map arithmetic.
 *
 * In plain words: a raster is a grid of numbers plus a second grid saying
 * which of those numbers are real ("valid"). Nodata is never turned into 0 --
 * a pixel is either valid with its value, or invalid and reads as `null`.
 *
 * Transforms use the GDAL / rasterio order `[a, b, c, d, e, f]`:
 *
 *     x = a * col + b * row + c
 *     y = d * col + e * row + f
 *
 * where (col, row) are continuous pixel coordinates with (0, 0) at the
 * top-left CORNER of the top-left pixel; the centre of pixel (i, j) is
 * (j + 0.5, i + 0.5). This is the same convention the Python pipeline writes
 * into its reports (`tuple(transform)[:6]`).
 */

/**
 * @typedef {'dsm'|'dtm'|'ndsm'|'agl'|'ground_mask'|'error'|'confidence'|'image'|'truth_height'|'relative_height'|'unknown'} ProductType
 *
 * @typedef {object} Raster
 * @property {number} width
 * @property {number} height
 * @property {ArrayLike<number>} values   row-major, original dtype preserved
 * @property {Uint8Array} valid           1 = real value, 0 = nodata / non-finite
 * @property {string} dtype
 * @property {number|null} nodata         the value declared in the file, or null
 * @property {string|null} crs            e.g. "EPSG:32643"; null when the source has none
 * @property {number[]|null} transform    [a,b,c,d,e,f]; null when the source has none
 * @property {string|null} units          e.g. "metres"; null when not declared
 * @property {ProductType} productType
 * @property {'ABSOLUTE'|'RELATIVE'|null} mode   null when not a height-like product
 * @property {string} modeReason
 * @property {Record<string,string>} tags dataset-level metadata tags
 * @property {boolean} synthetic
 * @property {string} source             where it was loaded from
 * @property {string} format             "GeoTIFF" | "NPY"
 * @property {string[]} warnings
 */

export class RasterError extends Error {}

/**
 * Validity from values and declared nodata: finite, and not equal to nodata.
 * Float32 values are compared against the float32-rounded nodata, which is
 * what the writer actually stored.
 * @param {ArrayLike<number>} values
 * @param {number|null} nodata
 * @returns {Uint8Array}
 */
export function validityMask(values, nodata) {
  const valid = new Uint8Array(values.length);
  const hasNodata = nodata !== null && nodata !== undefined && !Number.isNaN(nodata);
  const nd32 = hasNodata ? Math.fround(nodata) : NaN;
  const isF32 = values instanceof Float32Array;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (hasNodata && (v === nodata || (isF32 && v === nd32))) continue;
    valid[i] = 1;
  }
  return valid;
}

/**
 * Assemble and sanity-check a raster record.
 * @param {Partial<Raster> & {width:number,height:number,values:ArrayLike<number>,valid:Uint8Array}} fields
 * @returns {Raster}
 */
export function makeRaster(fields) {
  const { width, height, values, valid } = fields;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RasterError(`invalid raster size ${width} x ${height}`);
  }
  if (values.length !== width * height) {
    throw new RasterError(`values has ${values.length} entries, expected ${width} x ${height} = ${width * height}`);
  }
  if (valid.length !== values.length) {
    throw new RasterError(`validity mask has ${valid.length} entries, expected ${values.length}`);
  }
  if (fields.transform != null && fields.transform.length !== 6) {
    throw new RasterError(`transform must have 6 coefficients, got ${fields.transform.length}`);
  }
  return {
    dtype: 'unknown',
    nodata: null,
    crs: null,
    transform: null,
    units: null,
    productType: 'unknown',
    mode: null,
    modeReason: '',
    tags: {},
    synthetic: false,
    source: '',
    format: '',
    warnings: [],
    ...fields,
  };
}

/**
 * Value at an integer pixel, or null when outside the raster or invalid.
 * Never returns a substitute value.
 * @param {Raster} raster
 * @param {number} row
 * @param {number} col
 * @returns {number|null}
 */
export function valueAt(raster, row, col) {
  if (!Number.isInteger(row) || !Number.isInteger(col)) return null;
  if (row < 0 || col < 0 || row >= raster.height || col >= raster.width) return null;
  const i = row * raster.width + col;
  return raster.valid[i] ? Number(raster.values[i]) : null;
}

/**
 * Continuous pixel coordinates -> map coordinates.
 * @param {number[]} t  [a,b,c,d,e,f]
 * @param {number} col
 * @param {number} row
 * @returns {[number, number]}
 */
export function pixelToMap(t, col, row) {
  return [t[0] * col + t[1] * row + t[2], t[3] * col + t[4] * row + t[5]];
}

/**
 * Map coordinates -> continuous pixel coordinates (inverse affine).
 * @param {number[]} t  [a,b,c,d,e,f]
 * @param {number} x
 * @param {number} y
 * @returns {[number, number]}  [col, row]
 */
export function mapToPixel(t, x, y) {
  const [a, b, c, d, e, f] = t;
  const det = a * e - b * d;
  if (det === 0) throw new RasterError('transform is singular');
  const dx = x - c;
  const dy = y - f;
  return [(e * dx - b * dy) / det, (-d * dx + a * dy) / det];
}

/**
 * Integer pixel containing a map coordinate (floor, as rasterio.rowcol does),
 * or null outside the raster.
 * @param {Raster} raster
 * @param {number} x
 * @param {number} y
 * @returns {{row:number,col:number}|null}
 */
export function mapToPixelIndex(raster, x, y) {
  if (!raster.transform) throw new RasterError(`${raster.source || 'raster'} has no transform`);
  const [c, r] = mapToPixel(raster.transform, x, y);
  const col = Math.floor(c);
  const row = Math.floor(r);
  if (row < 0 || col < 0 || row >= raster.height || col >= raster.width) return null;
  return { row, col };
}

/**
 * True when two rasters share size, CRS and transform (tolerance 1e-9, the
 * same tolerance the Python validators use).
 * @param {Raster} a
 * @param {Raster} b
 * @returns {boolean}
 */
export function sameGrid(a, b) {
  if (a.width !== b.width || a.height !== b.height) return false;
  if ((a.crs ?? null) !== (b.crs ?? null)) return false;
  if ((a.transform == null) !== (b.transform == null)) return false;
  if (a.transform && b.transform) {
    for (let k = 0; k < 6; k++) if (Math.abs(a.transform[k] - b.transform[k]) > 1e-9) return false;
  }
  return true;
}

/**
 * Counts and range over valid pixels only.
 * @param {Raster} raster
 * @returns {{validPixels:number,nodataPixels:number,min:number|null,max:number|null}}
 */
export function validStats(raster) {
  let n = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < raster.values.length; i++) {
    if (!raster.valid[i]) continue;
    const v = Number(raster.values[i]);
    n++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return {
    validPixels: n,
    nodataPixels: raster.values.length - n,
    min: n ? min : null,
    max: n ? max : null,
  };
}

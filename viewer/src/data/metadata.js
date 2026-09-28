/**
 * Metadata interpretation: CRS, affine transform, product type, ABSOLUTE vs
 * RELATIVE mode, and how a value may be printed.
 *
 * In plain words: this file decides what a raster *is* (DSM? nDSM? error
 * map?) and whether its numbers are metres. It only reads what the file or
 * the pipeline's reports declare; it never guesses units.
 *
 * Mode rule (project contract, `src/depthwizard/mode.py`):
 *   ABSOLUTE  -- the raster has a CRS, an affine transform, and declares
 *                UNITS = metres. Values are printed in metres.
 *   RELATIVE  -- anything else. Values are unitless and are NEVER printed
 *                with "m".
 */

/** Product types whose values are heights (or height errors) and so have a mode. */
export const HEIGHT_LIKE = new Set(['dsm', 'dtm', 'ndsm', 'agl', 'error', 'truth_height', 'relative_height', 'unknown']);

/** What each product means, in the words the pipeline uses. */
export const PRODUCT_MEANINGS = {
  dsm: 'DSM: absolute surface elevation',
  dtm: 'DTM: absolute bare-ground elevation',
  ndsm: 'nDSM = DSM - DTM: object height above the extracted ground (Phase 6)',
  agl: 'AGL: Phase 3/4a model-derived height above ground (NOT the Phase 6 nDSM)',
  ground_mask: 'Phase 6 inferred ground mask: 1 ground, 0 non-ground',
  error: 'Phase 5 error = prediction - reference (positive = over-prediction)',
  confidence: 'Phase 5 confidence proxy: rule-based flags, NOT a probability',
  image: 'source image',
  truth_height: 'SYNTHETIC specified building height (fixture truth)',
  relative_height: 'Phase 3 relative height (unitless)',
  unknown: 'unknown product',
};

/**
 * CRS string from parsed GeoKeys, or null when the file declares none.
 * @param {Record<string, any>|null} geoKeys
 * @returns {string|null}
 */
export function crsFromGeoKeys(geoKeys) {
  if (!geoKeys) return null;
  const code = geoKeys.ProjectedCSTypeGeoKey ?? geoKeys.GeographicTypeGeoKey ?? null;
  if (code == null) {
    // A model type without an EPSG code is a user-defined CRS: present, but not
    // expressible here. Record that rather than dropping it.
    return geoKeys.GTModelTypeGeoKey != null ? 'user-defined' : null;
  }
  return code === 32767 ? 'user-defined' : `EPSG:${code}`;
}

/**
 * Affine transform [a,b,c,d,e,f] from raw GeoTIFF tags, following GDAL.
 *
 * - ModelPixelScale + ModelTiepoint (north-up), or ModelTransformation (may
 *   be rotated). Only the single-tiepoint form is supported.
 * - PixelIsPoint rasters are shifted by half a pixel so that (0,0) is the
 *   corner of the first pixel, exactly as GDAL/rasterio report them.
 *
 * @param {{pixelScale?:number[]|null, tiepoint?:number[]|null, modelTransformation?:number[]|null, pixelIsPoint?:boolean}} tags
 * @returns {number[]|null}
 */
export function affineFromTiffTags({ pixelScale = null, tiepoint = null, modelTransformation = null, pixelIsPoint = false }) {
  let t = null;
  if (modelTransformation && modelTransformation.length >= 8) {
    const m = modelTransformation;
    t = [m[0], m[1], m[3], m[4], m[5], m[7]];
  } else if (pixelScale && tiepoint && tiepoint.length === 6) {
    const [i, j, , x, y] = tiepoint;
    const [sx, sy] = pixelScale;
    t = [sx, 0, x - i * sx, 0, -sy, y + j * sy];
  } else if (tiepoint && tiepoint.length > 6) {
    throw new Error('multiple GCP tiepoints are not an affine transform; not supported');
  }
  if (t && pixelIsPoint) {
    t = [t[0], t[1], t[2] - 0.5 * t[0] - 0.5 * t[1], t[3], t[4], t[5] - 0.5 * t[3] - 0.5 * t[4]];
  }
  return t;
}

/**
 * Product type declared by the file's own tags (the pipeline writes CONTENT).
 * @param {Record<string,string>} tags
 * @returns {import('./raster.js').ProductType}
 */
export function inferProductType(tags) {
  const content = String(tags.CONTENT ?? '').trim();
  if ('STATE_0' in tags || /confidence proxy/i.test(content)) return 'confidence';
  if (/^nDSM\b/.test(content)) return 'ndsm';
  if (/^DSM\b/.test(content)) return 'dsm';
  if (/^DTM\b/.test(content)) return 'dtm';
  if (/^AGL\b/.test(content)) return 'agl';
  if (/^inferred ground mask/i.test(content)) return 'ground_mask';
  if (/^error\s*=/.test(content)) return 'error';
  if (/^ground_truth_building_height/.test(content)) return 'truth_height';
  return 'unknown';
}

/**
 * Reconcile the type a caller expects (e.g. from a report) with what the file
 * declares. A conflict is an error: a mislabelled product is worse than none.
 * @param {string|undefined} expected
 * @param {string} declared
 * @param {string} source
 * @returns {import('./raster.js').ProductType}
 */
export function reconcileProductType(expected, declared, source) {
  if (!expected) return /** @type {any} */ (declared);
  if (declared !== 'unknown' && declared !== expected) {
    throw new Error(`${source}: expected a ${expected} product but the file declares ${declared}`);
  }
  return /** @type {any} */ (expected);
}

/**
 * ABSOLUTE or RELATIVE, with the reason. Non-height products get null.
 * @param {{crs:string|null, transform:number[]|null, units:string|null, productType:string}} r
 * @returns {{mode:'ABSOLUTE'|'RELATIVE'|null, reason:string}}
 */
export function determineMode({ crs, transform, units, productType }) {
  if (!HEIGHT_LIKE.has(productType)) {
    return { mode: null, reason: `not applicable: ${productType} is not a height product` };
  }
  const missing = [];
  if (!crs) missing.push('no CRS');
  if (!transform) missing.push('no geotransform');
  if (!isMetres(units)) missing.push(units ? `units are "${units}", not metres` : 'no declared units');
  if (missing.length === 0) {
    return { mode: 'ABSOLUTE', reason: `georeferenced (${crs}) with declared units = metres` };
  }
  return { mode: 'RELATIVE', reason: `values are unitless: ${missing.join(', ')}` };
}

/** @param {string|null|undefined} units */
export function isMetres(units) {
  return typeof units === 'string' && /^(m|metre|metres|meter|meters)$/i.test(units.trim());
}

/**
 * The unit suffix a value may be printed with. RELATIVE is never metres.
 * @param {{mode:string|null, units:string|null}} raster
 * @returns {string}  " m" or "" (and the caller says "unitless")
 */
export function unitSuffix(raster) {
  return raster.mode === 'ABSOLUTE' && isMetres(raster.units) ? ' m' : '';
}

/**
 * Format a raster value for display. null -> "nodata".
 * @param {number|null} value
 * @param {{mode:string|null, units:string|null}} raster
 * @param {number} [digits=2]
 * @returns {string}
 */
export function formatValue(value, raster, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'nodata';
  const suffix = unitSuffix(raster);
  if (suffix) return `${value.toFixed(digits)}${suffix}`;
  return `${value.toFixed(Math.max(digits, 4))} (unitless, relative)`;
}

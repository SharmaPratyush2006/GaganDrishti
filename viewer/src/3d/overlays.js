/**
 * Phase 5 diagnostic overlays (error map, confidence flags) on the terrain.
 *
 * In plain words: a diagnostic raster is coloured pixel by pixel and blended
 * onto the terrain texture. It is shown only where it genuinely describes
 * the surface being displayed:
 *
 *   - Phase 5 validated the SOURCE DSM (its `prediction_path`). Phase 6
 *     re-exports that DSM as dsm.tif (tag SOURCE_DSM). So the diagnostics are
 *     shown on the DSM surface of that product, and nowhere else. On the nDSM
 *     or DTM they are refused with the reason -- a DSM error is not an nDSM
 *     error, and a DSM quality flag is not an nDSM quality flag.
 *   - The diagnostic raster must sit on exactly the displayed grid (size,
 *     transform, CRS -- texture.js textureAlignment); otherwise refused.
 *
 * Error map (float, "error = prediction - reference", positive = over-prediction):
 *   diverging blue - white - red, symmetric about 0, so the sign is never
 *   lost: negative is always blue-ish, positive always red-ish, 0 white.
 *   Colour saturates at +-L, L = 99th percentile of |error| over valid
 *   pixels (reported; values beyond are clamped in colour only).
 *
 * Confidence (uint8 state codes): the Phase 5 rule-based flags
 *   (validation/confidence.py; state table from the Phase 5 report,
 *   cross-checked against the file's STATE_<code> tags). Categorical colours,
 *   one per state. These are NOT probabilities and nothing here converts
 *   them into a number.
 *
 * Invalid pixels (declared nodata, non-finite, undocumented code) get alpha 0:
 * the terrain shows through, and they are counted. No replacement value.
 *
 * Texels follow texture.js: texel (row, col) = raster pixel (row, col), and
 * every LOD tile samples the one full-raster texture through its own UVs, so
 * the overlay follows whatever tiles are selected, with no shift.
 */
import { isMetres } from '../data/metadata.js';
import { NODATA_TEXEL, textureAlignment } from './texture.js';

export const CONFIDENCE_DISCLAIMER = 'Rule-based quality flags, not probabilities.';
export const ERROR_LABEL = 'Phase 5 DSM error — prediction minus reference';
export const CONFIDENCE_LABEL = 'Phase 5 DSM confidence proxy — rule-based quality state';
export const ERROR_SATURATION_PERCENTILE = 99;
export const DEFAULT_OVERLAY_OPACITY = 0.65;

const SURFACE_NAMES = { ndsm: 'nDSM', dtm: 'DTM', dsm: 'DSM', relative_height: 'Phase 3 relative raster' };

/** Colour per documented state NAME (the code comes from the report, never assumed). */
export const STATE_COLORS = {
  INVALID: [60, 60, 60],
  UNSUITABLE: [123, 50, 148],
  REDUCED: [253, 174, 97],
  NOT_ASSESSED: [116, 173, 209],
  HIGH: [26, 152, 80],
};
const FALLBACK_STATE_COLORS = [[240, 240, 60], [230, 90, 200], [160, 120, 60]];

// Diverging stops (ColorBrewer RdBu ends + neutral centre).
const NEG = [33, 102, 172];
const MID = [247, 247, 247];
const POS = [178, 24, 43];

const which = (kind) => (kind === 'error' ? 'Error' : 'Confidence');

/**
 * Whether a Phase 5 diagnostic may be drawn on the displayed surface.
 * @param {'error'|'confidence'} kind
 * @param {{surfaceRaster: import('../data/raster.js').Raster, diagnostic: import('../data/raster.js').Raster|null|undefined,
 *   products?: object|null, notAvailable?: Record<string,string>}} ctx
 * @returns {{available:boolean, reason:string|null, alignment:object|null}}
 */
export function diagnosticEligibility(kind, { surfaceRaster, diagnostic, products = null, notAvailable = {} }) {
  const name = which(kind);
  const surface = surfaceRaster.productType;
  const shown = SURFACE_NAMES[surface] ?? surface;
  const no = (reason, alignment = null) => ({ available: false, reason, alignment });

  if (surface !== 'dsm') {
    const base = kind === 'error'
      ? `Error overlay unavailable for this ${shown} — Phase 5 error map validates the source DSM, not ${shown}.`
      : `Confidence overlay unavailable for this ${shown} — available only for the Phase 5 source DSM; not ${surface === 'ndsm' ? 'an' : 'a'} ${shown} confidence map.`;
    const hint = diagnostic ? ' Switch the surface to DSM to view it.' : '';
    const extra = !diagnostic && notAvailable[kind] ? ` (Also: ${notAvailable[kind]}.)` : '';
    return no(base + hint + extra);
  }
  if (!diagnostic) {
    return no(`${name} overlay unavailable — no aligned ${kind === 'error' ? 'error' : 'confidence'} raster: ${notAvailable[kind] ?? 'none linked for this product'}`);
  }
  if (diagnostic.productType !== kind) return no(`${name} overlay unavailable — linked file is a ${diagnostic.productType} product, not ${kind}`);
  if (kind === 'error' && diagnostic.tags?.PRODUCT && diagnostic.tags.PRODUCT !== 'dsm') {
    return no(`Error overlay unavailable — the error map validates a ${diagnostic.tags.PRODUCT} product, not a DSM`);
  }
  // The displayed DSM must be the very DSM Phase 5 validated.
  const declared = surfaceRaster.tags?.SOURCE_DSM;
  if (declared && products?.sourceDsm && norm(declared) !== norm(products.sourceDsm)) {
    return no(`${name} overlay unavailable — the displayed DSM was exported from ${norm(declared)}, but Phase 5 validated ${norm(products.sourceDsm)}`);
  }
  const alignment = textureAlignment(surfaceRaster, diagnostic);
  if (!alignment.aligned) {
    return no(`${name} overlay unavailable — no aligned ${kind} raster (${alignment.reasons.join('; ')})`, alignment);
  }
  return { available: true, reason: null, alignment };
}

const norm = (p) => String(p).replace(/\\/g, '/');

/** p-th percentile (nearest rank) of |value| over valid, finite pixels. */
export function absPercentile(raster, p) {
  const abs = [];
  for (let i = 0; i < raster.values.length; i++) {
    const v = Number(raster.values[i]);
    if (raster.valid[i] && Number.isFinite(v)) abs.push(Math.abs(v));
  }
  if (!abs.length) return null;
  const sorted = Float64Array.from(abs).sort();
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))];
}

/**
 * Signed error -> RGB on the diverging ramp. t in [-1, 1] after clamping.
 * @returns {[number,number,number]}
 */
export function divergingColor(value, limit) {
  const t = limit > 0 ? Math.max(-1, Math.min(1, value / limit)) : 0;
  const [from, to, s] = t < 0 ? [MID, NEG, -t] : [MID, POS, t];
  return [0, 1, 2].map((k) => Math.round(from[k] + (to[k] - from[k]) * s));
}

/**
 * Error raster -> RGBA overlay texels.
 * @param {import('../data/raster.js').Raster} err
 */
export function errorOverlayData(err) {
  const { width, height, values, valid } = err;
  const limit = absPercentile(err, ERROR_SATURATION_PERCENTILE);
  const data = new Uint8Array(width * height * 4);
  let invalid = 0;
  let positive = 0;
  let negative = 0;
  let zero = 0;
  let saturated = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < width * height; i++) {
    const v = Number(values[i]);
    if (!valid[i] || !Number.isFinite(v)) {
      invalid++;
      continue; // alpha 0: nothing drawn
    }
    if (v > 0) positive++;
    else if (v < 0) negative++;
    else zero++;
    if (limit !== null && Math.abs(v) > limit) saturated++;
    if (v < min) min = v;
    if (v > max) max = v;
    data.set(divergingColor(v, limit ?? 0), i * 4);
    data[i * 4 + 3] = 255;
  }
  const n = positive + negative + zero;
  return {
    width, height, data,
    limit, min: n ? min : null, max: n ? max : null,
    counts: { positive, negative, zero, invalid, saturated },
  };
}

/**
 * The state table actually used: the Phase 5 report's, cross-checked against
 * the raster's STATE_<code> tags. A disagreement is an error, not a guess.
 * @param {Record<string,{code:number,meaning:string}>|null|undefined} reportStates
 * @param {Record<string,string>} tags
 * @returns {{states: {code:number,name:string,meaning:string,color:number[]}[]}|{error:string}}
 */
export function confidenceStateTable(reportStates, tags = {}) {
  const fromTags = Object.entries(tags)
    .map(([k, v]) => [/^STATE_(\d+)$/.exec(k), v])
    .filter(([m]) => m)
    .map(([m, name]) => ({ code: Number(m[1]), name: String(name) }));
  let states;
  if (reportStates && Object.keys(reportStates).length) {
    states = Object.entries(reportStates).map(([name, s]) => ({ code: Number(s.code), name, meaning: s.meaning ?? '' }));
    for (const t of fromTags) {
      const r = states.find((s) => s.code === t.code);
      if (!r || r.name !== t.name) {
        return { error: `the raster tags STATE_${t.code}=${t.name} but the Phase 5 report says ${r ? r.name : 'nothing'} for code ${t.code}` };
      }
    }
  } else if (fromTags.length) {
    states = fromTags.map((t) => ({ ...t, meaning: '(meaning not recorded in the report)' }));
  } else {
    return { error: 'no documented state table (neither the Phase 5 report nor the raster tags define the codes)' };
  }
  const codes = new Set();
  for (const s of states) {
    if (!Number.isInteger(s.code) || s.code < 0 || s.code > 255 || codes.has(s.code)) return { error: `invalid or duplicate state code ${s.code}` };
    codes.add(s.code);
  }
  let fb = 0;
  states.sort((a, b) => a.code - b.code);
  return { states: states.map((s) => ({ ...s, color: STATE_COLORS[s.name] ?? FALLBACK_STATE_COLORS[fb++ % FALLBACK_STATE_COLORS.length] })) };
}

/**
 * Confidence raster -> RGBA overlay texels, one colour per documented state.
 * @param {import('../data/raster.js').Raster} conf
 * @param {{code:number,name:string,color:number[]}[]} states
 */
export function confidenceOverlayData(conf, states) {
  const { width, height, values, valid } = conf;
  const byCode = new Map(states.map((s) => [s.code, s]));
  const data = new Uint8Array(width * height * 4);
  const counts = Object.fromEntries(states.map((s) => [s.name, 0]));
  let nodata = 0;
  let undocumented = 0;
  for (let i = 0; i < width * height; i++) {
    if (!valid[i]) {
      nodata++;
      continue;
    }
    const s = byCode.get(Number(values[i]));
    if (!s) {
      undocumented++;
      continue;
    }
    counts[s.name]++;
    data.set(s.color, i * 4);
    data[i * 4 + 3] = 255;
  }
  return { width, height, data, counts, nodata, undocumented };
}

/**
 * Blend overlay texels onto base texels. Base null -> the untextured surface
 * colour. Overlay alpha 0 leaves the base untouched.
 * @param {Uint8Array|null} base  RGBA
 * @param {Uint8Array} overlay    RGBA
 * @param {number} opacity        0..1
 */
export function compositeTexels(base, overlay, opacity) {
  const out = new Uint8Array(overlay.length);
  for (let o = 0; o < overlay.length; o += 4) {
    const a = (overlay[o + 3] / 255) * opacity;
    for (let k = 0; k < 3; k++) {
      const b = base ? base[o + k] : NODATA_TEXEL[k];
      out[o + k] = Math.round(b + (overlay[o + k] - b) * a);
    }
    out[o + 3] = 255;
  }
  return out;
}

/**
 * Everything the UI needs for one overlay kind on one displayed surface.
 * @param {'error'|'confidence'} kind
 * @param {Parameters<typeof diagnosticEligibility>[1] & {confidenceStates?: object|null}} ctx
 */
export function buildOverlay(kind, ctx) {
  const t0 = performance.now();
  const elig = diagnosticEligibility(kind, ctx);
  const d = ctx.diagnostic;
  const base = { kind, available: false, reason: elig.reason, alignment: elig.alignment, source: d?.source ?? null };
  if (!elig.available) return base;
  if (kind === 'error') {
    const ov = errorOverlayData(d);
    if (ov.limit === null) return { ...base, reason: 'Error overlay unavailable — the error raster has no valid pixels' };
    return {
      ...base, available: true, label: ERROR_LABEL,
      meaning: 'positive (red) = DSM above the reference (over-prediction); negative (blue) = below (under-prediction)',
      units: d.mode === 'ABSOLUTE' && isMetres(d.units) ? 'm' : 'unitless',
      overlay: ov, buildMs: performance.now() - t0,
    };
  }
  const table = confidenceStateTable(ctx.confidenceStates, d.tags);
  if (table.error) return { ...base, reason: `Confidence overlay unavailable — ${table.error}` };
  const ov = confidenceOverlayData(d, table.states);
  return {
    ...base, available: true, label: CONFIDENCE_LABEL, disclaimer: CONFIDENCE_DISCLAIMER,
    states: table.states, overlay: ov, buildMs: performance.now() - t0,
  };
}

/**
 * The overlay's value at a pixel, as recorded (no interpolation, no conversion).
 * @returns {{text:string, value:number|null, state?:string}|null}
 */
export function overlayValueAt(ov, raster, row, col) {
  if (!ov?.available || !raster) return null;
  if (row < 0 || col < 0 || row >= raster.height || col >= raster.width) return null;
  const i = row * raster.width + col;
  const v = Number(raster.values[i]);
  if (!raster.valid[i] || !Number.isFinite(v)) return { text: 'nodata', value: null };
  if (ov.kind === 'error') {
    const sign = v > 0 ? '+' : '';
    return { text: `${sign}${v.toFixed(3)}${ov.units === 'm' ? ' m' : ' (unitless)'}`, value: v };
  }
  const s = ov.states.find((st) => st.code === v);
  return s ? { text: `${s.name} (code ${v})`, value: v, state: s.name } : { text: `undocumented code ${v}`, value: v };
}

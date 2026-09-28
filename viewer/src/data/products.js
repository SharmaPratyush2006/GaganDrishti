/**
 * Which files belong together, resolved only from the pipeline's own reports.
 *
 * In plain words: nothing here knows a file name in advance. It asks the dev
 * server which reports exist, then follows the links the reports record:
 *
 *   phase6_report.json
 *     products[<name>].exports.{dsm,dtm,ndsm,ground_mask}.tif .path  -> the surfaces
 *     products[<name>].source_dsm                                   -> the Phase 4b DSM it came from
 *     reproducibility.inputs.phase4b_report                         -> phase4b_report.json
 *   phase4b_report.json
 *     image_grid.source                                             -> the source image
 *   phase5_report.json   (any, whose products[i].prediction_path == source_dsm)
 *     products[i].error_map.raster.path, products[i].confidence_raster
 *
 * The Phase 5 products describe the SOURCE DSM's validation, not the Phase 6
 * nDSM's accuracy; they are labelled that way here, once, for every consumer.
 * A link that cannot be followed becomes an explicit "not available" reason.
 */
import { normalizeReportPath, reportPathToUrl, samePath } from './paths.js';

export const PHASE5_SOURCE_LABEL = 'Phase 5 source DSM diagnostic (validation of the DSM this nDSM was derived from; NOT nDSM accuracy)';

const SURFACE_FILES = { dsm: 'dsm.tif', dtm: 'dtm.tif', ndsm: 'ndsm.tif', ground_mask: 'ground_mask.tif' };

/** @param {string} path */
export function reportPhase(path) {
  const m = /(?:^|\/)phase(\d+[a-z]?)_report\.json$/.exec(normalizeReportPath(path));
  return m ? m[1] : null;
}

/**
 * @typedef {{path:string, json:any}} LoadedReport
 * @typedef {{url:string, path:string, productType:string, label?:string}} FileRef
 * @typedef {object} DemoProducts
 * @property {string} phase6Report
 * @property {string} productName
 * @property {string} role
 * @property {string[]} availableProducts
 * @property {string} label
 * @property {boolean} synthetic
 * @property {Record<string,string>} definitions
 * @property {string} sourceDsm
 * @property {Record<string, FileRef>} surfaces
 * @property {FileRef|null} image
 * @property {FileRef|null} errorMap
 * @property {FileRef|null} confidence
 * @property {Record<string,{code:number,meaning:string}>|null} confidenceStates
 * @property {Record<string,string>} notAvailable   product -> reason
 */

/**
 * Resolve one Phase 6 product and everything linked to it.
 * @param {LoadedReport[]} reports
 * @param {{phase6Report?:string, productName?:string}} [choice]
 * @returns {DemoProducts}
 */
export function resolveDemoProducts(reports, { phase6Report, productName } = {}) {
  const byPhase = (phase) => reports.filter((r) => reportPhase(r.path) === phase);
  const phase6s = byPhase('6');
  if (phase6s.length === 0) throw new Error('no phase6_report.json found under data/outputs (run Phase 6 first)');
  let r6;
  if (phase6Report) {
    r6 = phase6s.find((r) => samePath(r.path, phase6Report));
    if (!r6) throw new Error(`Phase 6 report not found: ${phase6Report}`);
  } else if (phase6s.length === 1) {
    r6 = phase6s[0];
  } else {
    throw new Error(`several Phase 6 reports found; choose one: ${phase6s.map((r) => r.path).join(', ')}`);
  }
  const j6 = r6.json;
  const available = Object.keys(j6.products ?? {});
  const name = productName ?? j6.acceptance?.input;
  const product = j6.products?.[name];
  if (!product) throw new Error(`${r6.path}: product ${JSON.stringify(name)} not in ${available.join(', ')}`);

  const notAvailable = {};
  const surfaces = {};
  for (const [type, file] of Object.entries(SURFACE_FILES)) {
    const path = product.exports?.[file]?.path;
    if (path && product.exports[file].ok !== false) {
      surfaces[type] = { url: reportPathToUrl(path), path: normalizeReportPath(path), productType: type };
    } else {
      notAvailable[type] = `${r6.path} records no validated ${file} for ${name}`;
    }
  }

  // Source image via the Phase 4b report that Phase 6 read.
  let image = null;
  const p4bPath = j6.reproducibility?.inputs?.phase4b_report;
  const r4b = p4bPath ? reports.find((r) => samePath(r.path, p4bPath)) : undefined;
  const imagePath = r4b?.json?.image_grid?.source;
  if (imagePath) {
    image = { url: reportPathToUrl(imagePath), path: normalizeReportPath(imagePath), productType: 'image' };
  } else {
    notAvailable.image = p4bPath
      ? (r4b ? `${p4bPath} records no image_grid.source` : `Phase 4b report ${p4bPath} is not under data/outputs`)
      : `${r6.path} does not record its Phase 4b report`;
  }

  // Phase 5 products whose prediction is exactly this source DSM.
  const sourceDsm = product.source_dsm;
  const matches = [];
  for (const r5 of byPhase('5')) {
    for (const p of r5.json.products ?? []) {
      if (p.prediction_path && sourceDsm && samePath(p.prediction_path, sourceDsm)) matches.push({ r5, p });
    }
  }
  let errorMap = null;
  let confidence = null;
  let confidenceStates = null;
  if (matches.length > 1) {
    const where = matches.map((m) => m.r5.path).join(', ');
    notAvailable.error = notAvailable.confidence = `ambiguous: several Phase 5 reports validated ${sourceDsm} (${where})`;
  } else if (matches.length === 0) {
    // Say what Phase 5 DID validate, and which Phase 6 product that corresponds to.
    const validated = [];
    for (const r5 of byPhase('5')) {
      for (const p of r5.json.products ?? []) {
        if (!p.prediction_path) continue;
        const owners = available.filter((n) => j6.products[n].source_dsm && samePath(j6.products[n].source_dsm, p.prediction_path));
        if (owners.length) validated.push(`${normalizeReportPath(p.prediction_path)} (Phase 6 product ${owners.join(', ')})`);
      }
    }
    const where = validated.length
      ? `; Phase 5 diagnostics exist only for ${validated.join('; ')}, not for this product`
      : '; no Phase 5 report validated any source DSM of this Phase 6 run';
    notAvailable.error = notAvailable.confidence =
      `no Phase 5 report validated this product's source DSM (${sourceDsm ? normalizeReportPath(sourceDsm) : 'none recorded'})${where}`;
  } else {
    const { r5, p } = matches[0];
    const errPath = p.error_map?.raster?.path;
    if (errPath) {
      errorMap = { url: reportPathToUrl(errPath), path: normalizeReportPath(errPath), productType: 'error', label: PHASE5_SOURCE_LABEL };
    } else {
      notAvailable.error = `${r5.path} records no error map raster for ${sourceDsm}`;
    }
    if (p.confidence_raster) {
      confidence = { url: reportPathToUrl(p.confidence_raster), path: normalizeReportPath(p.confidence_raster), productType: 'confidence', label: PHASE5_SOURCE_LABEL };
      confidenceStates = p.confidence?.states ?? null;
    } else {
      notAvailable.confidence = `${r5.path} records no confidence raster for ${sourceDsm}`;
    }
  }
  // Never available from any phase: stated, not filled.
  notAvailable.uncertainty = 'no per-pixel uncertainty product exists in the pipeline';

  return {
    phase6Report: normalizeReportPath(r6.path),
    productName: name,
    role: product.role ?? '',
    availableProducts: available,
    label: j6.label ?? '',
    synthetic: /SYNTHETIC/.test(j6.label ?? ''),
    definitions: j6.definitions ?? {},
    sourceDsm: sourceDsm ? normalizeReportPath(sourceDsm) : '',
    surfaces,
    image,
    errorMap,
    confidence,
    confidenceStates,
    notAvailable,
  };
}

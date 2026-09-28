/**
 * Fetch + decode: turns resolved product links into Raster records.
 *
 * In plain words: ask the dev server what reports exist, read them, resolve
 * the linked files (products.js), download and decode each raster, then check
 * that every optional layer sits on exactly the same grid as the nDSM. A
 * layer that is missing or misaligned is reported, never stretched to fit.
 *
 * `fetchImpl` is injectable so the same code runs under tests.
 */
import { readGeoTiff } from './geotiffLoader.js';
import { readNpyRaster } from './npy.js';
import { reportPathToUrl } from './paths.js';
import { reportPhase, resolveDemoProducts } from './products.js';
import { sameGrid } from './raster.js';

export const INDEX_URL = '/__data_index';
const LINKED_PHASES = new Set(['4b', '5', '6']);

async function fetchOk(fetchImpl, url) {
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`GET ${url} failed: HTTP ${res.status}`);
  return res;
}

/** @returns {Promise<{root:string, reports:string[], npy:string[]}>} */
export async function fetchIndex(fetchImpl = fetch) {
  return (await fetchOk(fetchImpl, INDEX_URL)).json();
}

/** @returns {Promise<import('./products.js').LoadedReport[]>} */
export async function fetchReports(paths, fetchImpl = fetch) {
  return Promise.all(paths.map(async (path) => ({ path, json: await (await fetchOk(fetchImpl, reportPathToUrl(path))).json() })));
}

/**
 * @param {string} url
 * @param {{productType?:string, fetchImpl?:typeof fetch}} [options]
 */
export async function fetchGeoTiff(url, { productType, fetchImpl = fetch } = {}) {
  const buffer = await (await fetchOk(fetchImpl, url)).arrayBuffer();
  // Source images keep every band so their real channel layout can be used.
  return readGeoTiff(buffer, { source: url, productType, allBands: productType === 'image' });
}

/** @param {string} path  repo-relative .npy path from the index */
export async function fetchNpy(path, fetchImpl = fetch) {
  const url = reportPathToUrl(path);
  const buffer = await (await fetchOk(fetchImpl, url)).arrayBuffer();
  return readNpyRaster(buffer, { source: url });
}

/**
 * Load one Phase 6 product and its linked layers.
 * @param {{phase6Report?:string, productName?:string}} [choice]
 * @param {typeof fetch} [fetchImpl]
 */
export async function loadDemo(choice = {}, fetchImpl = fetch) {
  const t0 = performance.now();
  const index = await fetchIndex(fetchImpl);
  // Only the reports that carry the links products.js follows.
  const reports = await fetchReports(index.reports.filter((p) => LINKED_PHASES.has(reportPhase(p))), fetchImpl);
  const products = resolveDemoProducts(reports, choice);

  /** @type {Record<string, import('./raster.js').Raster>} */
  const rasters = {};
  const notAvailable = { ...products.notAvailable };
  const entries = [
    ...Object.values(products.surfaces),
    ...[products.image, products.errorMap, products.confidence].filter(Boolean),
  ];
  await Promise.all(entries.map(async (ref) => {
    try {
      rasters[ref.productType] = await fetchGeoTiff(ref.url, { productType: ref.productType, fetchImpl });
    } catch (err) {
      notAvailable[ref.productType] = `failed to load ${ref.path}: ${err.message}`;
    }
  }));

  const reference = rasters.ndsm;
  const alignment = {};
  if (reference) {
    for (const [type, r] of Object.entries(rasters)) {
      if (type === 'ndsm') continue;
      alignment[type] = sameGrid(reference, r);
      if (!alignment[type]) {
        r.warnings.push('NOT on the nDSM grid (size, CRS or transform differ): not aligned, not used as a layer');
      }
    }
  } else {
    notAvailable.ndsm ??= 'nDSM could not be loaded';
  }

  return { index, products, rasters, alignment, notAvailable, loadMs: performance.now() - t0 };
}

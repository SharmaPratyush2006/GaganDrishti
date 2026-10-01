/**
 * Phase 8: which demo manifest and which relative .npy to load, chosen explicitly.
 *
 * In plain words: the viewer never picks a file by index order. The demo
 * manifest (demo/reports/demo_report.json) names its own Phase 6 report and
 * product; a .npy is loaded only when it was chosen, or is the only one.
 * Anything ambiguous or missing is an error with the reason, never a guess.
 *
 * `fetchImpl` is injectable so the same code runs under tests.
 */
import { fetchIndex, loadDemo } from './load.js';
import { reportPathToUrl } from './paths.js';

/** @param {string} p  repo-relative report path */
export const isDemoManifest = (p) => /(?:^|\/)demo_report\.json$/.test(p);

/** The Phase 3 relative .npy the "Load a Phase 3 relative .npy" button uses when none is chosen. */
export const DEFAULT_PHASE3_NPY = 'data/outputs/phase3/verification/JAX_004_006_r256_c256_relative_height.npy';

/**
 * The .npy to load: the chosen one, or the only one in the index.
 * @param {{npy:string[]}} index
 * @param {string} [chosen]
 */
export function chooseNpy(index, chosen) {
  if (index.npy.length === 0) throw new Error('no .npy relative-height output found under data/outputs');
  const path = chosen || (index.npy.length === 1 ? index.npy[0] : null);
  if (!path) throw new Error(`several .npy relative-height outputs found; choose one: ${index.npy.join(', ')}`);
  if (!index.npy.includes(path)) throw new Error(`.npy not found under data/outputs: ${path}`);
  return path;
}

/**
 * What to load at start-up, from the URL: ?demo=1 or the bare root URL -> the Phase 8 demo;
 * otherwise an explicit ?phase6Report= or ?npy=; any other query loads nothing.
 * @param {URLSearchParams} params
 * @returns {'demo'|'phase6'|'npy'|null}
 */
export function startupLoad(params) {
  if (params.get('demo') === '1') return 'demo';
  if (params.get('phase6Report')) return 'phase6';
  if (params.get('npy')) return 'npy';
  return params.toString() === '' ? 'demo' : null;
}

/**
 * Read the one Phase 8 demo manifest and the Phase 6 choice it records.
 * @param {typeof fetch} [fetchImpl]
 * @returns {Promise<{manifest:any, path:string, phase6Report:string, productName:string|undefined}>}
 */
export async function readDemoManifest(fetchImpl = fetch) {
  const idx = await fetchIndex(fetchImpl);
  const demos = idx.reports.filter(isDemoManifest);
  if (demos.length === 0) throw new Error('no Phase 8 demo found under data/outputs (run `python -m depthwizard.demo synthetic`)');
  if (demos.length > 1) throw new Error(`several Phase 8 demo manifests found: ${demos.join(', ')}`);
  const res = await fetchImpl(reportPathToUrl(demos[0]));
  if (!res.ok) throw new Error(`GET ${demos[0]} failed: HTTP ${res.status}`);
  const manifest = await res.json();
  const phase6Report = manifest.viewer?.phase6_report;
  if (!phase6Report) throw new Error(`${demos[0]} records no viewer.phase6_report`);
  return { manifest, path: demos[0], phase6Report, productName: manifest.phase6_surfaces?.acceptance_input };
}

/**
 * Load the Phase 6 product the demo manifest names (never another Phase 6 run).
 * @param {typeof fetch} [fetchImpl]
 */
export async function loadPhase8Demo(fetchImpl = fetch) {
  const { manifest, path, phase6Report, productName } = await readDemoManifest(fetchImpl);
  const loaded = await loadDemo({ phase6Report, productName }, fetchImpl);
  return { ...loaded, demo: { manifest, path } };
}

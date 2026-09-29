/**
 * Vite dev-server plugin: read-only access to the pipeline's outputs.
 *
 * In plain words: the browser cannot read files from disk by itself, and
 * `data/` lives outside `viewer/`. This serves exactly one directory,
 * `<repo>/data/outputs`, at `/data/outputs/*` -- GET/HEAD only, no directory
 * listings, no path escaping -- plus `/__data_index`, a list of the pipeline
 * reports (and the Phase 8 demo manifest) and .npy arrays that exist, so the
 * viewer never hard-codes a path.
 *
 * Development-only: Vite binds to localhost by default.
 */
import fs from 'node:fs';
import path from 'node:path';

const URL_PREFIX = '/data/outputs/';
const INDEX_URL = '/__data_index';
// Pipeline reports, plus the Phase 8 demo manifest (demo/reports/demo_report.json).
const REPORT_RE = /^(?:phase\d+[a-z]?|demo)_report\.json$/;
const MAX_DEPTH = 6;
const CONTENT_TYPES = {
  '.tif': 'image/tiff',
  '.tiff': 'image/tiff',
  '.json': 'application/json; charset=utf-8',
  '.npy': 'application/octet-stream',
  '.png': 'image/png',
  '.md': 'text/markdown; charset=utf-8',
};

/**
 * Resolve a URL path to a file inside `outputsDir`, or null if it escapes.
 * @param {string} outputsDir absolute
 * @param {string} urlPath    e.g. "/data/outputs/phase6/x.tif"
 */
export function resolveDataPath(outputsDir, urlPath) {
  if (!urlPath.startsWith(URL_PREFIX)) return null;
  let rel;
  try {
    rel = decodeURIComponent(urlPath.slice(URL_PREFIX.length));
  } catch {
    return null;
  }
  if (rel.includes('\0') || rel.split(/[\\/]/).includes('..')) return null;
  const full = path.resolve(outputsDir, rel);
  const inside = path.relative(outputsDir, full);
  if (!inside || inside.startsWith('..') || path.isAbsolute(inside)) return null;
  return full;
}

/**
 * Walk `outputsDir` for reports and .npy arrays (repo-relative POSIX paths).
 * @param {string} outputsDir
 */
export function buildIndex(outputsDir) {
  const reports = [];
  const npy = [];
  const walk = (dir, depth) => {
    if (depth > MAX_DEPTH) return;
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) walk(full, depth + 1);
      else if (e.isFile()) {
        const rel = 'data/outputs/' + path.relative(outputsDir, full).split(path.sep).join('/');
        if (REPORT_RE.test(e.name)) reports.push(rel);
        else if (e.name.endsWith('.npy')) npy.push(rel);
      }
    }
  };
  walk(outputsDir, 0);
  return { root: 'data/outputs', reports: reports.sort(), npy: npy.sort() };
}

/**
 * Connect-style middleware.
 * @param {{outputsDir:string}} options
 */
export function createDataHandler({ outputsDir }) {
  const root = path.resolve(outputsDir);
  return function dataHandler(req, res, next) {
    const urlPath = (req.url ?? '').split('?')[0];
    if (urlPath !== INDEX_URL && !urlPath.startsWith(URL_PREFIX)) return next();
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.statusCode = 405;
      res.setHeader('Allow', 'GET, HEAD');
      return res.end('read-only');
    }
    if (urlPath === INDEX_URL) {
      const body = JSON.stringify(buildIndex(root));
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.setHeader('Cache-Control', 'no-store');
      return res.end(req.method === 'HEAD' ? undefined : body);
    }
    const file = resolveDataPath(root, urlPath);
    if (!file) {
      res.statusCode = 403;
      return res.end('forbidden');
    }
    let stat;
    try {
      stat = fs.statSync(file);
    } catch {
      stat = null;
    }
    if (!stat || !stat.isFile()) {
      res.statusCode = 404;
      return res.end('not found');
    }
    res.setHeader('Content-Type', CONTENT_TYPES[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
    res.setHeader('Content-Length', String(stat.size));
    res.setHeader('Cache-Control', 'no-store');
    if (req.method === 'HEAD') return res.end();
    fs.createReadStream(file).pipe(res);
  };
}

/**
 * @param {{outputsDir:string}} options
 * @returns {import('vite').Plugin}
 */
export function depthwizardData({ outputsDir }) {
  const handler = createDataHandler({ outputsDir });
  return {
    name: 'depthwizard-data',
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

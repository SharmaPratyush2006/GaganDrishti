/**
 * Report path -> URL on the viewer's read-only data route.
 *
 * In plain words: the Python reports store paths like
 * `data\outputs\phase6\synthetic\synthetic_dsm\ndsm.tif` (repo-relative,
 * Windows separators). The dev server serves exactly `data/outputs/` at
 * `/data/outputs/`, so a report path maps to a URL by fixing the separators.
 * Anything outside `data/outputs/`, absolute, or containing `..` is refused
 * rather than guessed at.
 */

export const DATA_PREFIX = 'data/outputs/';

export class PathError extends Error {}

/**
 * Normalise a report path to a repo-relative POSIX path.
 * @param {string} p
 * @returns {string}
 */
export function normalizeReportPath(p) {
  if (typeof p !== 'string' || p.trim() === '') throw new PathError(`not a path: ${JSON.stringify(p)}`);
  let s = p.trim().replace(/\\/g, '/').replace(/\/{2,}/g, '/');
  while (s.startsWith('./')) s = s.slice(2);
  if (/^[A-Za-z]:\//.test(s) || s.startsWith('/')) {
    throw new PathError(`absolute path in report is not supported (expected repo-relative): ${p}`);
  }
  if (s.split('/').some((seg) => seg === '..')) throw new PathError(`path escapes the repository: ${p}`);
  return s;
}

/**
 * URL for a report path on the data route.
 * @param {string} p
 * @returns {string}
 */
export function reportPathToUrl(p) {
  const s = normalizeReportPath(p);
  if (!s.startsWith(DATA_PREFIX)) throw new PathError(`not under ${DATA_PREFIX} (not served): ${p}`);
  return '/' + s.split('/').map(encodeURIComponent).join('/');
}

/**
 * True when two report paths name the same file.
 * @param {string} a
 * @param {string} b
 */
export function samePath(a, b) {
  return normalizeReportPath(a) === normalizeReportPath(b);
}

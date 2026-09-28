import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildIndex, resolveDataPath } from '../server/dataRoute.js';

export const TESTS_DIR = path.dirname(fileURLToPath(import.meta.url));
export const FIXTURES = path.join(TESTS_DIR, 'fixtures');
export const REPO_ROOT = path.resolve(TESTS_DIR, '..', '..');
export const OUTPUTS_DIR = path.join(REPO_ROOT, 'data', 'outputs');

/** @param {string} file */
export function readArrayBuffer(file) {
  const b = fs.readFileSync(file);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

export const expected = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'expected.json'), 'utf-8'));

/** Real pipeline outputs are git-ignored; tests on them skip when absent (as the Python suite does). */
export function hasPhase6Outputs() {
  return buildIndex(OUTPUTS_DIR).reports.some((p) => p.endsWith('phase6_report.json'));
}

/**
 * A fetch() that answers exactly as the dev-server data route would, from disk.
 * Uses the route's own path resolution and index.
 */
export function diskFetch(outputsDir = OUTPUTS_DIR) {
  const respond = (status, body) => ({
    ok: status === 200,
    status,
    json: async () => JSON.parse(body.toString('utf-8')),
    arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength),
  });
  return async (url) => {
    if (url === '/__data_index') return respond(200, Buffer.from(JSON.stringify(buildIndex(outputsDir))));
    const file = resolveDataPath(outputsDir, url);
    if (!file) return respond(403, Buffer.from('forbidden'));
    if (!fs.existsSync(file)) return respond(404, Buffer.from('not found'));
    return respond(200, fs.readFileSync(file));
  };
}

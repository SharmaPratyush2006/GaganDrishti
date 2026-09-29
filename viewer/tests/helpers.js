import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildIndex } from '../server/dataRoute.js';
import { diskFetch as serverDiskFetch } from '../server/diskFetch.js';

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

/**
 * The Phase 6 run these real-data tests were written against. Named explicitly
 * because data/outputs may hold several Phase 6 reports (e.g. the Phase 8 demo's),
 * and the loader rightly refuses to guess between them.
 */
export const PHASE6_REPORT = 'data/outputs/phase6/synthetic/phase6_report.json';

/** Real pipeline outputs are git-ignored; tests on them skip when absent (as the Python suite does). */
export function hasPhase6Outputs() {
  return buildIndex(OUTPUTS_DIR).reports.includes(PHASE6_REPORT);
}

/**
 * A fetch() that answers exactly as the dev-server data route would, from disk.
 * Uses the route's own path resolution and index (server/diskFetch.js).
 */
export function diskFetch(outputsDir = OUTPUTS_DIR) {
  return serverDiskFetch(outputsDir);
}

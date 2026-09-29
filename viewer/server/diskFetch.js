/**
 * A fetch() that answers exactly as the dev-server data route would, from disk.
 *
 * In plain words: lets the viewer's own loading code (load.js, products.js)
 * run under Node -- in the tests and in scripts/export-mesh.mjs -- with the
 * route's own path resolution and index, so nothing is resolved differently
 * from the browser.
 */
import fs from 'node:fs';

import { buildIndex, resolveDataPath } from './dataRoute.js';

/** @param {string} outputsDir absolute path of `<repo>/data/outputs` */
export function diskFetch(outputsDir) {
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

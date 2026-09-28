import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Writable } from 'node:stream';

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { buildIndex, createDataHandler, resolveDataPath } from '../server/dataRoute.js';

let tmp;
let outputs;

beforeAll(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'dw-route-'));
  outputs = path.join(tmp, 'data', 'outputs');
  fs.mkdirSync(path.join(outputs, 'phase6', 'synthetic'), { recursive: true });
  fs.mkdirSync(path.join(outputs, 'phase3'), { recursive: true });
  fs.writeFileSync(path.join(outputs, 'phase6', 'synthetic', 'phase6_report.json'), '{"phase":"6"}');
  fs.writeFileSync(path.join(outputs, 'phase6', 'synthetic', 'ndsm.tif'), 'TIFFDATA');
  fs.writeFileSync(path.join(outputs, 'phase3', 'x_relative_height.npy'), 'NPY');
  fs.writeFileSync(path.join(tmp, 'data', 'secret.txt'), 'outside outputs');
});
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

/** Drive the middleware with a fake request/response. */
function request(method, url) {
  return new Promise((resolve) => {
    const chunks = [];
    const headers = {};
    const res = new Writable({
      write(chunk, _enc, cb) { chunks.push(Buffer.from(chunk)); cb(); },
    });
    res.statusCode = 200;
    res.setHeader = (k, v) => { headers[k.toLowerCase()] = v; };
    const origEnd = res.end.bind(res);
    res.end = (body) => {
      if (body) chunks.push(Buffer.from(body));
      origEnd();
      resolve({ status: res.statusCode, headers, body: Buffer.concat(chunks).toString(), next: false });
      return res;
    };
    createDataHandler({ outputsDir: outputs })({ method, url }, res, () => resolve({ next: true }));
  });
}

describe('data route', () => {
  it('serves files under data/outputs', async () => {
    const r = await request('GET', '/data/outputs/phase6/synthetic/ndsm.tif');
    expect(r.status).toBe(200);
    expect(r.body).toBe('TIFFDATA');
    expect(r.headers['content-type']).toBe('image/tiff');
  });

  it('refuses path traversal (raw and encoded)', async () => {
    expect((await request('GET', '/data/outputs/../secret.txt')).status).toBe(403);
    expect((await request('GET', '/data/outputs/%2e%2e/secret.txt')).status).toBe(403);
    expect((await request('GET', '/data/outputs/..%5Csecret.txt')).status).toBe(403);
    expect(resolveDataPath(outputs, '/data/outputs/')).toBeNull();
  });

  it('is read-only', async () => {
    expect((await request('POST', '/data/outputs/phase6/synthetic/ndsm.tif')).status).toBe(405);
    expect((await request('PUT', '/__data_index')).status).toBe(405);
  });

  it('404s missing files and does not list directories', async () => {
    expect((await request('GET', '/data/outputs/nope.tif')).status).toBe(404);
    expect((await request('GET', '/data/outputs/phase6')).status).toBe(404);
  });

  it('passes other URLs through to Vite', async () => {
    expect((await request('GET', '/src/main.jsx')).next).toBe(true);
  });

  it('indexes reports and .npy arrays with repo-relative POSIX paths', async () => {
    const r = await request('GET', '/__data_index');
    expect(JSON.parse(r.body)).toEqual(buildIndex(outputs));
    expect(buildIndex(outputs)).toEqual({
      root: 'data/outputs',
      reports: ['data/outputs/phase6/synthetic/phase6_report.json'],
      npy: ['data/outputs/phase3/x_relative_height.npy'],
    });
  });
});

import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { formatValue } from '../src/data/metadata.js';
import { parseNpy, readNpyRaster } from '../src/data/npy.js';
import { valueAt } from '../src/data/raster.js';
import { OUTPUTS_DIR, readArrayBuffer } from './helpers.js';
import { buildIndex } from '../server/dataRoute.js';

/** Build a .npy v1.0 file the way numpy.save does (header padded to 64 bytes). */
function npy(descr, shape, bytes, { fortran = false } = {}) {
  const shapeText = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let header = `{'descr': '${descr}', 'fortran_order': ${fortran ? 'True' : 'False'}, 'shape': ${shapeText}, }`;
  const pad = 64 - ((10 + header.length + 1) % 64);
  header += ' '.repeat(pad % 64) + '\n';
  const out = new Uint8Array(10 + header.length + bytes.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(new TextEncoder().encode(header), 10);
  out.set(new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength), 10 + header.length);
  return out.buffer;
}

describe('parseNpy / readNpyRaster', () => {
  it('reads float32 C-order with shape and NaN as invalid', () => {
    const r = readNpyRaster(npy('<f4', [2, 3], new Float32Array([0, 1, NaN, 3, 4, 5])), { source: 't.npy' });
    expect([r.width, r.height]).toEqual([3, 2]);
    expect(valueAt(r, 0, 0)).toBe(0);
    expect(valueAt(r, 0, 2)).toBeNull();
    expect(valueAt(r, 1, 2)).toBe(5);
  });

  it('is always RELATIVE, unitless, with no CRS or transform', () => {
    const r = readNpyRaster(npy('<f4', [2, 2], new Float32Array([1, 2, 3, 4])));
    expect(r.mode).toBe('RELATIVE');
    expect(r.units).toBeNull();
    expect(r.crs).toBeNull();
    expect(r.transform).toBeNull();
    expect(r.productType).toBe('relative_height');
    expect(r.format).toBe('NPY');
    expect(formatValue(valueAt(r, 1, 1), r)).not.toMatch(/\bm\b|metre|meter/);
  });

  it('transposes Fortran order into row-major', () => {
    // Logical [[1,2,3],[4,5,6]] stored column-major: 1,4,2,5,3,6
    const r = readNpyRaster(npy('<f8', [2, 3], new Float64Array([1, 4, 2, 5, 3, 6]), { fortran: true }));
    expect(Array.from(r.values)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('reads big-endian data', () => {
    const buf = new ArrayBuffer(8);
    const dv = new DataView(buf);
    dv.setFloat32(0, 1.5, false);
    dv.setFloat32(4, -2.25, false);
    expect(Array.from(parseNpy(npy('>f4', [1, 2], new Uint8Array(buf))).data)).toEqual([1.5, -2.25]);
  });

  it('squeezes (1, H, W) to 2-D and rejects true 3-D arrays', () => {
    expect(readNpyRaster(npy('|u1', [1, 2, 2], new Uint8Array([1, 2, 3, 4]))).height).toBe(2);
    expect(() => readNpyRaster(npy('|u1', [2, 2, 2], new Uint8Array(8)))).toThrow(/expected a 2-D array/);
  });

  it('rejects bad magic, unsupported dtypes and truncated data', () => {
    expect(() => parseNpy(new ArrayBuffer(16))).toThrow(/bad magic/);
    expect(() => parseNpy(npy('<c8', [1], new Uint8Array(8)))).toThrow(/unsupported .npy dtype/);
    expect(() => parseNpy(npy('<f4', [4], new Float32Array(2)))).toThrow(/truncated/);
  });
});

const phase3Npy = buildIndex(OUTPUTS_DIR).npy.find((p) => p.includes('relative_height'));

describe.skipIf(!phase3Npy)('real Phase 3 relative-height .npy', () => {
  it('loads as a RELATIVE 2-D raster matching the file header', () => {
    const file = path.join(OUTPUTS_DIR, '..', '..', phase3Npy);
    const r = readNpyRaster(readArrayBuffer(file), { source: phase3Npy });
    const header = fs.readFileSync(file).subarray(10, 128).toString('latin1');
    const [h, w] = /'shape': \((\d+), (\d+)\)/.exec(header).slice(1).map(Number);
    expect([r.height, r.width]).toEqual([h, w]);
    expect(r.mode).toBe('RELATIVE');
    expect(r.dtype).toBe('float32');
  });
});

/**
 * NumPy .npy -> Raster, for Phase 3 relative-height outputs.
 *
 * In plain words: Phase 3 saves its relative height field as a .npy array. It
 * has no CRS, no transform and no units, so it is always RELATIVE: the values
 * only say "higher / lower", and they are never printed as metres.
 *
 * Kept separate from the GeoTIFF loader on purpose: the two formats share
 * only the Raster record they produce.
 *
 * Supported: format versions 1.0 / 2.0 / 3.0; dtypes f4, f8, u1, i1, u2, i2,
 * u4, i4, b1 in either byte order; C or Fortran order; shapes that squeeze to
 * 2-D (e.g. (1, H, W)). Non-finite values are invalid; there is no nodata tag.
 */
import { determineMode } from './metadata.js';
import { makeRaster, RasterError, validityMask } from './raster.js';

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

const DTYPES = {
  f4: { size: 4, name: 'float32', Array: Float32Array, get: 'getFloat32' },
  f8: { size: 8, name: 'float64', Array: Float64Array, get: 'getFloat64' },
  u1: { size: 1, name: 'uint8', Array: Uint8Array, get: 'getUint8' },
  i1: { size: 1, name: 'int8', Array: Int8Array, get: 'getInt8' },
  b1: { size: 1, name: 'bool', Array: Uint8Array, get: 'getUint8' },
  u2: { size: 2, name: 'uint16', Array: Uint16Array, get: 'getUint16' },
  i2: { size: 2, name: 'int16', Array: Int16Array, get: 'getInt16' },
  u4: { size: 4, name: 'uint32', Array: Uint32Array, get: 'getUint32' },
  i4: { size: 4, name: 'int32', Array: Int32Array, get: 'getInt32' },
};

/**
 * Parse the header and data of a .npy file.
 * @param {ArrayBuffer} buffer
 * @returns {{shape:number[], dtype:string, fortranOrder:boolean, data:ArrayLike<number>}}
 */
export function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10 || MAGIC.some((b, i) => bytes[i] !== b)) {
    throw new RasterError('not a .npy file (bad magic string)');
  }
  const major = bytes[6];
  const view = new DataView(buffer);
  let headerLen;
  let offset;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    offset = 10;
  } else if (major === 2 || major === 3) {
    headerLen = view.getUint32(8, true);
    offset = 12;
  } else {
    throw new RasterError(`unsupported .npy format version ${major}`);
  }
  const header = new TextDecoder(major === 3 ? 'utf-8' : 'latin1').decode(bytes.subarray(offset, offset + headerLen));
  const dataStart = offset + headerLen;

  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)?.[1];
  const shapeText = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || !fortran || shapeText === undefined) throw new RasterError(`unparseable .npy header: ${header.trim()}`);

  const order = descr[0];
  const kind = descr.slice(1);
  const spec = DTYPES[kind];
  if (!spec || !'<>|='.includes(order)) throw new RasterError(`unsupported .npy dtype ${descr}`);
  const shape = shapeText.split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  if (dataStart + count * spec.size > buffer.byteLength) {
    throw new RasterError(`.npy data is truncated: need ${count * spec.size} bytes after the header`);
  }

  let data;
  const littleEndian = order !== '>';
  if (littleEndian || spec.size === 1) {
    // slice() copies, so the typed array is always correctly aligned.
    data = new spec.Array(buffer.slice(dataStart, dataStart + count * spec.size));
  } else {
    data = new spec.Array(count);
    for (let i = 0; i < count; i++) data[i] = view[spec.get](dataStart + i * spec.size, false);
  }
  return { shape, dtype: spec.name, fortranOrder: fortran === 'True', data };
}

/**
 * Read a .npy as a RELATIVE, unitless raster.
 * @param {ArrayBuffer} buffer
 * @param {{source?:string}} [options]
 * @returns {import('./raster.js').Raster}
 */
export function readNpyRaster(buffer, { source = '' } = {}) {
  const { shape, dtype, fortranOrder, data } = parseNpy(buffer);
  const dims = shape.filter((d) => d !== 1);
  if (shape.length < 2 || dims.length > 2) {
    throw new RasterError(`${source}: expected a 2-D array, got shape (${shape.join(', ')})`);
  }
  const height = shape.length === 2 ? shape[0] : (dims.length === 2 ? dims[0] : 1);
  const width = shape.length === 2 ? shape[1] : (dims.length >= 1 ? dims[dims.length - 1] : 1);

  let values = data;
  if (fortranOrder && height > 1 && width > 1) {
    values = new data.constructor(width * height);
    for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) values[r * width + c] = data[c * height + r];
  }
  const productType = 'relative_height';
  const { mode, reason } = determineMode({ crs: null, transform: null, units: null, productType });
  return makeRaster({
    width, height, values, valid: validityMask(values, null), dtype, nodata: null,
    crs: null, transform: null, units: null, productType, mode, modeReason: reason,
    tags: {}, synthetic: false, source, format: 'NPY', warnings: [],
  });
}

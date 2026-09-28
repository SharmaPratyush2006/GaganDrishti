/**
 * GeoTIFF / COG -> Raster, via geotiff.js.
 *
 * In plain words: give it the bytes of a .tif (from fetch, or later from a
 * file picker) and it returns the full-resolution band, a validity mask, and
 * the georeferencing and tags exactly as the file declares them.
 *
 * Decisions:
 * - Only the first image (full resolution) is read; COG overviews are ignored.
 * - Nodata comes from the GDAL_NODATA tag. Pixels equal to it, or non-finite,
 *   are invalid. Nothing is replaced by 0.
 * - geotiff.js fills *sparse* blocks (byte count 0) with `nodata || 0`, which
 *   invents zeros when there is no nodata (or nodata is NaN). Those blocks are
 *   marked invalid here explicitly.
 * - The affine transform is computed from the raw tags (metadata.js), not
 *   from geotiff.js's getResolution(), whose rotated-case handling differs from GDAL.
 */
import { fromArrayBuffer } from 'geotiff';

import {
  affineFromTiffTags,
  crsFromGeoKeys,
  determineMode,
  inferProductType,
  reconcileProductType,
} from './metadata.js';
import { makeRaster, RasterError, validityMask } from './raster.js';

const SAMPLE_FORMATS = { 1: 'uint', 2: 'int', 3: 'float' };

async function tagValue(fd, name) {
  return fd.hasTag(name) ? await fd.loadValue(name) : null;
}

function toArray(v) {
  return v == null ? null : Array.from(v, Number);
}

/**
 * Mark pixels of blocks stored with zero bytes as invalid.
 * @returns {number} number of sparse blocks found
 */
async function invalidateSparseBlocks(image, valid, width, height) {
  const fd = image.getFileDirectory();
  const counts = toArray(await tagValue(fd, image.isTiled ? 'TileByteCounts' : 'StripByteCounts'));
  if (!counts) return 0;
  const bw = image.getTileWidth();
  const bh = image.isTiled ? image.getTileHeight() : Number((await tagValue(fd, 'RowsPerStrip')) ?? height);
  const perRow = Math.ceil(width / bw);
  const blocksPerBand = perRow * Math.ceil(height / bh);
  let sparse = 0;
  // Band 0 only (planar configuration 2 stores later bands after it).
  for (let i = 0; i < Math.min(blocksPerBand, counts.length); i++) {
    if (counts[i] !== 0) continue;
    sparse++;
    const r0 = Math.floor(i / perRow) * bh;
    const c0 = (i % perRow) * bw;
    for (let r = r0; r < Math.min(r0 + bh, height); r++) {
      valid.fill(0, r * width + c0, r * width + Math.min(c0 + bw, width));
    }
  }
  return sparse;
}

/**
 * Decode a single band of a GeoTIFF.
 * @param {ArrayBuffer} buffer
 * Every result records the file's band structure (`bands`, `bandDtypes`,
 * `photometric`, `extraSamples`) so an image's real channel layout is known.
 * With `allBands`, every band is also read, pixel-interleaved, into `pixels`
 * (used for source images; `values`/`valid` remain the selected band).
 *
 * @param {ArrayBuffer} buffer
 * @param {{source?:string, productType?:string, band?:number, allBands?:boolean}} [options]
 * @returns {Promise<import('./raster.js').Raster>}
 */
export async function readGeoTiff(buffer, { source = '', productType, band = 0, allBands = false } = {}) {
  let tiff;
  try {
    tiff = await fromArrayBuffer(buffer);
  } catch (err) {
    throw new RasterError(`${source}: not a readable TIFF (${err.message})`);
  }
  const image = await tiff.getImage(0);
  const width = image.getWidth();
  const height = image.getHeight();
  const bands = image.getSamplesPerPixel();
  if (band < 0 || band >= bands) throw new RasterError(`${source}: band ${band} requested, file has ${bands}`);

  const fd = image.getFileDirectory();
  const formats = toArray(await tagValue(fd, 'SampleFormat')) ?? [];
  const bitsList = toArray(await tagValue(fd, 'BitsPerSample')) ?? [];
  const bandDtypes = Array.from({ length: bands }, (_, k) => `${SAMPLE_FORMATS[formats[k] ?? formats[0] ?? 1] ?? 'unknown'}${bitsList[k] ?? bitsList[0]}`);
  const dtype = bandDtypes[band];
  const photometric = (await tagValue(fd, 'PhotometricInterpretation')) ?? null;
  const extraSamples = toArray(await tagValue(fd, 'ExtraSamples')) ?? [];

  const [values] = await image.readRasters({ samples: [band] });
  const pixels = allBands ? await image.readRasters({ interleave: true }) : null;
  const nodata = image.getGDALNoData();
  const valid = validityMask(values, nodata);
  const warnings = [];
  const sparse = await invalidateSparseBlocks(image, valid, width, height);
  if (sparse) warnings.push(`${sparse} sparse (unwritten) block(s) treated as nodata`);

  const tags = /** @type {Record<string,string>} */ ((await image.getGDALMetadata()) ?? {});
  const geoKeys = image.getGeoKeys();
  const crs = crsFromGeoKeys(geoKeys);
  let transform = affineFromTiffTags({
    pixelScale: toArray(await tagValue(fd, 'ModelPixelScale')),
    tiepoint: toArray(await tagValue(fd, 'ModelTiepoint')),
    modelTransformation: toArray(await tagValue(fd, 'ModelTransformation')),
    pixelIsPoint: geoKeys?.GTRasterTypeGeoKey === 2,
  });
  const isIdentity = transform && transform.every((v, k) => v === [1, 0, 0, 0, 1, 0][k]);
  if (isIdentity && !crs) {
    // Pixel-space grid (e.g. DFC2019 tiles): there is no georeferencing to carry.
    transform = null;
    warnings.push('identity transform with no CRS: pixel-space grid, not georeferenced');
  }

  const declared = inferProductType(tags);
  const type = reconcileProductType(productType, declared, source);
  const units = tags.UNITS ?? null;
  const { mode, reason } = determineMode({ crs, transform, units, productType: type });

  return makeRaster({
    width, height, values, valid, dtype, nodata, crs, transform, units,
    productType: type, mode, modeReason: reason, tags,
    synthetic: String(tags.SYNTHETIC ?? '').toLowerCase() === 'true',
    source, format: 'GeoTIFF', warnings,
    bands, bandDtypes, photometric: photometric === null ? null : Number(photometric), extraSamples, pixels,
  });
}

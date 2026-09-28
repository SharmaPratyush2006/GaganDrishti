/**
 * Source image -> terrain texture, only when the image sits on the same grid.
 *
 * In plain words: the source image is draped over the terrain so that image
 * pixel (row, col) lies exactly on height sample (row, col). That is only
 * true when both rasters share size, transform and CRS, so anything else is
 * refused with a reason instead of being stretched to fit.
 *
 * What the image is, is read from the file, never assumed:
 *   grayscale -- 1 band, PhotometricInterpretation 1 (BlackIsZero)
 *   rgb       -- >= 3 bands, PhotometricInterpretation 2 (RGB); bands 1-3 used
 *   anything else (multispectral without an RGB declaration, palette,
 *   WhiteIsZero, non-uint8 samples) is reported as unsupported.
 *
 * Texture layout (THREE.DataTexture, RGBA uint8, sRGB, flipY = false):
 *   texel row k = image row k, stored first-row-first. With flipY false the
 *   first stored row is at v = 0, and tileMesh.js gives sample (row, col)
 *   u = (col + 0.5) / W, v = (row + 0.5) / H: the centre of texel (row, col).
 *   So image row 0 lands on raster row 0 -- no vertical flip.
 *
 * A grayscale image is displayed with R = G = B = its value, which is simply
 * how a single-band image is shown; it is labelled grayscale, never RGB.
 * Image nodata texels are drawn in the untextured terrain colour and counted
 * (they carry no image data).
 */
import * as THREE from 'three';

import { BASE_COLOR } from './terrain.js';

/** Untextured terrain colour (terrain.js BASE_COLOR), for image nodata texels. */
export const NODATA_TEXEL = [(BASE_COLOR >> 16) & 255, (BASE_COLOR >> 8) & 255, BASE_COLOR & 255, 255];

/**
 * @param {import('../data/raster.js').Raster|null|undefined} img
 * @returns {{kind:'grayscale'|'rgb'|'unsupported'|'missing', label:string, channels:number, reason:string|null}}
 */
export function describeSourceImage(img) {
  if (!img) return { kind: 'missing', label: 'no source image', channels: 0, reason: 'no source image is linked for this raster' };
  const bands = img.bands ?? 1;
  const photometric = img.photometric ?? null;
  const dtypes = img.bandDtypes ?? [img.dtype];
  if (!img.pixels) return { kind: 'unsupported', label: 'unsupported', channels: bands, reason: 'image bands were not loaded (allBands)' };
  if (dtypes.some((d) => d !== 'uint8')) {
    return { kind: 'unsupported', label: `unsupported (${dtypes.join('/')})`, channels: bands,
      reason: 'only uint8 images are drawn; other sample types would need a stretch that alters values' };
  }
  if (bands === 1 && photometric === 1) return { kind: 'grayscale', label: 'grayscale (1 band)', channels: 1, reason: null };
  if (bands >= 3 && photometric === 2) {
    const extra = bands > 3 ? `; ${bands - 3} extra band(s) not drawn` : '';
    return { kind: 'rgb', label: `RGB (${bands} bands${extra})`, channels: bands, reason: null };
  }
  return {
    kind: 'unsupported', label: `unsupported (${bands} band(s), photometric ${photometric})`, channels: bands,
    reason: bands >= 3 ? 'several bands but the file does not declare them as RGB; not assumed to be RGB'
      : 'single band that is not BlackIsZero grayscale',
  };
}

/**
 * Whether an image may be draped on a height raster, and why not.
 * @param {import('../data/raster.js').Raster} heightRaster
 * @param {import('../data/raster.js').Raster} img
 * @returns {{aligned:boolean, reasons:string[]}}
 */
export function textureAlignment(heightRaster, img) {
  const reasons = [];
  if (img.width !== heightRaster.width || img.height !== heightRaster.height) {
    reasons.push(`size ${img.width}×${img.height} ≠ raster ${heightRaster.width}×${heightRaster.height}`);
  }
  if (!img.transform || !heightRaster.transform) {
    reasons.push('no transform on ' + (!img.transform && !heightRaster.transform ? 'either grid' : !img.transform ? 'the image' : 'the raster') + ' to verify alignment');
  } else if (img.transform.some((v, k) => Math.abs(v - heightRaster.transform[k]) > 1e-9)) {
    reasons.push(`transform [${img.transform.join(', ')}] ≠ raster [${heightRaster.transform.join(', ')}]`);
  }
  if ((img.crs ?? null) !== (heightRaster.crs ?? null)) {
    reasons.push(`CRS ${img.crs ?? 'none'} ≠ raster ${heightRaster.crs ?? 'none'}`);
  }
  return { aligned: reasons.length === 0, reasons };
}

/**
 * RGBA texel data from the image's real channels.
 * @param {import('../data/raster.js').Raster} img
 * @param {'grayscale'|'rgb'} kind
 * @returns {{width:number, height:number, data:Uint8Array, nodataTexels:number}}
 */
export function textureData(img, kind) {
  const { width, height, pixels, valid } = img;
  const bands = img.bands ?? 1;
  const n = width * height;
  const data = new Uint8Array(n * 4);
  let nodataTexels = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (!valid[i]) {
      data.set(NODATA_TEXEL, o);
      nodataTexels++;
      continue;
    }
    if (kind === 'grayscale') {
      const g = pixels[i];
      data[o] = g; data[o + 1] = g; data[o + 2] = g;
    } else {
      data[o] = pixels[i * bands]; data[o + 1] = pixels[i * bands + 1]; data[o + 2] = pixels[i * bands + 2];
    }
    data[o + 3] = 255;
  }
  return { width, height, data, nodataTexels };
}

/** @param {{width:number,height:number,data:Uint8Array}} td */
export function createDataTexture(td) {
  const tex = new THREE.DataTexture(td.data, td.width, td.height, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = false;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Everything the viewer needs to decide about the texture for one raster.
 * @param {import('../data/raster.js').Raster} heightRaster
 * @param {import('../data/raster.js').Raster|null|undefined} img
 */
export function textureForRaster(heightRaster, img) {
  const t0 = performance.now();
  const source = describeSourceImage(img);
  const base = { source, alignment: null, texture: null, nodataTexels: 0, buildMs: null };
  if (source.kind === 'missing') return { ...base, status: 'unavailable', label: `Texture unavailable — ${source.reason}` };
  const alignment = textureAlignment(heightRaster, img);
  if (!alignment.aligned) {
    return { ...base, alignment, status: 'unavailable',
      label: `Texture unavailable — source image is not aligned with the selected raster (${alignment.reasons.join('; ')})` };
  }
  if (source.kind === 'unsupported') return { ...base, alignment, status: 'unavailable', label: `Texture unavailable — ${source.reason}` };
  const td = textureData(img, source.kind);
  const texture = createDataTexture(td);
  return {
    ...base, alignment, texture, nodataTexels: td.nodataTexels, status: source.kind,
    label: source.kind === 'grayscale' ? 'Texture: grayscale source image' : 'Texture: RGB source image',
    buildMs: performance.now() - t0,
  };
}

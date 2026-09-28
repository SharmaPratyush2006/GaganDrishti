/**
 * Displayed raster -> binary glTF (.glb), optionally Draco-compressed.
 *
 * In plain words: the terrain is rebuilt ONCE at full resolution with the
 * viewer's own mesh code (heightmap.js + tileMesh.js: one node covering the
 * whole raster at stride 1), so every exported vertex is exactly the viewer's
 * sample: a valid pixel's centre (col+0.5, row+0.5) through the raster's
 * affine transform, re-centred on the raster centre. Nothing else builds
 * geometry here.
 *
 * What is exported
 *   - one mesh, full resolution (NOT the camera-dependent LOD tiles)
 *   - POSITION, NORMAL, TEXCOORD_0, UNSIGNED_INT indices
 *   - the aligned source image as a PNG base-colour texture (grayscale stays
 *     grayscale), else a plain grey material
 *   - metadata in asset.extras and the node's extras (see buildMetadata)
 * Not exported: Phase 5 overlays (diagnostics without their legend), the
 * linked cursor / measurement markers, uncertainty (no such product exists).
 *
 * Heights (y): ALWAYS the raster value itself (vertical scale 1).
 *   ABSOLUTE -> metres, 1:1 with the metre map axes.
 *   RELATIVE -> unitless relative values, NOT metres. The viewer draws these
 *     with a display-only exaggeration; that factor is recorded in the extras
 *     (displayedVerticalScale) but NOT applied, so the file holds the values.
 *
 * Nodata: invalid pixels never become vertices; a triangle is written only
 * when its 3 vertices are valid (tileMesh.js, stride 1). Holes stay open.
 *
 * Draco (KHR_draco_mesh_compression) is applied only when an encoder module
 * is supplied. Without one the export FAILS rather than writing an
 * uncompressed file labelled Draco. Every result is re-parsed
 * (inspectGlb) and `draco.verified` is true only if the file really carries
 * the extension on its primitive. Draco quantises attributes (lossy): the
 * bits used are recorded.
 */
import { Document, WebIO } from '@gltf-transform/core';
import { KHRDracoMeshCompression } from '@gltf-transform/extensions';

import { heightmapFromRaster } from '../3d/heightmap.js';
import { BASE_COLOR, tileGeometry } from '../3d/terrain.js';
import { describeSourceImage, textureAlignment, textureData } from '../3d/texture.js';
import { buildTileMesh } from '../3d/tileMesh.js';
import { HEIGHT_LIKE, isMetres, PRODUCT_MEANINGS } from '../data/metadata.js';
import { encodePng } from './png.js';

export const GENERATOR = 'DepthWizard viewer (Phase 7 Step 7 glTF export)';
export const DRACO_EXTENSION = 'KHR_draco_mesh_compression';
/** Quantisation bits per attribute (Draco is lossy; 16-bit positions ≈ extent / 65535). */
export const DRACO_QUANTIZATION = { POSITION: 16, NORMAL: 10, TEX_COORD: 14 };

export class ExportError extends Error {}

/**
 * The full-resolution export mesh, built with the viewer's own code.
 * @param {import('../data/raster.js').Raster} raster
 */
export function buildExportMesh(raster) {
  if (!raster) throw new ExportError('no raster is loaded');
  if (!HEIGHT_LIKE.has(raster.productType) || raster.mode === null) {
    throw new ExportError(`${raster.productType} is not a height product; nothing to export as terrain`);
  }
  // Values exactly (scale 1); the display scale is only reported.
  const hm = heightmapFromRaster(raster, { verticalScale: 1 });
  const displayed = heightmapFromRaster(raster);
  const node = { id: 'export', level: 0, r0: 0, c0: 0, r1: raster.height - 1, c1: raster.width - 1, stride: 1 };
  const tile = buildTileMesh(hm, node);
  if (tile.triangleCount === 0) throw new ExportError('the raster has no valid triangle to export (all nodata, or fewer than 2 × 2 valid pixels)');
  const geometry = tileGeometry(tile); // same normals as the viewer tiles
  const normals = Float32Array.from(geometry.attributes.normal.array);
  geometry.dispose();
  return { hm, displayed, tile, normals };
}

/** Texture for the export: the aligned source image, as the viewer drapes it. */
export async function buildExportTexture(raster, sourceImage) {
  const source = describeSourceImage(sourceImage);
  if (source.kind === 'missing') return { included: false, reason: source.reason };
  const alignment = textureAlignment(raster, sourceImage);
  if (!alignment.aligned) return { included: false, reason: `source image not aligned with the raster (${alignment.reasons.join('; ')})` };
  if (source.kind === 'unsupported') return { included: false, reason: source.reason };
  const td = textureData(sourceImage, source.kind);
  // Grayscale without nodata texels stays a 1-channel PNG; otherwise RGBA as displayed.
  const gray = source.kind === 'grayscale' && td.nodataTexels === 0;
  const data = gray ? td.data.filter((_, i) => i % 4 === 0) : td.data;
  const png = await encodePng({ width: td.width, height: td.height, data, format: gray ? 'gray' : 'rgba' });
  return {
    included: true, png, kind: source.kind, pngFormat: gray ? 'grayscale' : 'RGBA',
    label: source.kind === 'grayscale' ? 'grayscale source image (not RGB)' : 'RGB source image',
    nodataTexels: td.nodataTexels, source: sourceImage.source,
  };
}

/** Everything a consumer needs to interpret the file; values only from the raster and the viewer. */
export function buildMetadata(raster, mesh, texture, draco) {
  const { hm, displayed, tile } = mesh;
  const absolute = raster.mode === 'ABSOLUTE' && isMetres(raster.units);
  let invalid = 0;
  for (let i = 0; i < raster.valid.length; i++) if (!raster.valid[i]) invalid++;
  return {
    generator: GENERATOR,
    source: raster.source,
    productType: raster.productType,
    product: PRODUCT_MEANINGS[raster.productType] ?? raster.productType,
    synthetic: Boolean(raster.synthetic),
    width: raster.width,
    height: raster.height,
    crs: raster.crs ?? null,
    transform: raster.transform ?? null,
    transformConvention: 'GDAL/rasterio [a,b,c,d,e,f]: x = a*col + b*row + c, y = d*col + e*row + f; (col,row) = (0,0) at the top-left pixel corner',
    mode: raster.mode,
    modeReason: raster.modeReason,
    units: raster.units ?? null,
    heightValues: absolute
      ? 'y = raster value in metres (ABSOLUTE)'
      : 'y = raster value, RELATIVE and unitless — NOT metres',
    horizontalUnits: hm.horizontalUnits,
    verticalScaleApplied: 1,
    displayedVerticalScale: displayed.verticalScale,
    displayedVerticalScaleReason: displayed.verticalScaleReason,
    coordinateFrame: {
      axes: 'glTF Y-up: +x = map east, +y = up (raster value), -z = map north',
      origin: hm.origin,
      toMap: 'map x = x + origin[0]; map y = -z + origin[1]',
      vertex: 'one vertex per valid pixel centre (col + 0.5, row + 0.5) through the transform',
      recentred: 'origin = map coordinate of the raster centre, subtracted for float32 precision',
    },
    nodata: {
      value: raster.nodata ?? null,
      invalidPixels: invalid,
      trianglesOmitted: tile.droppedForNodata,
      rule: 'nodata / non-finite pixels are never vertices; a triangle is written only if all 3 vertices are valid; holes are left open, nothing is filled',
    },
    mesh: { resolution: 'full resolution (stride 1), not the viewer\'s camera-dependent LOD tiles', vertices: tile.vertexCount, triangles: tile.triangleCount },
    texture: texture.included
      ? { included: true, content: texture.label, png: texture.pngFormat, source: texture.source, nodataTexels: texture.nodataTexels, uv: 'texel centre of the same pixel: ((col+0.5)/W, (row+0.5)/H), first image row at v = 0' }
      : { included: false, reason: texture.reason },
    notIncluded: ['Phase 5 error/confidence overlays', 'linked-cursor and measurement markers', 'uncertainty (no per-pixel uncertainty product exists)'],
    compression: draco
      ? { draco: true, extension: DRACO_EXTENSION, lossy: true, quantizationBits: { ...DRACO_QUANTIZATION } }
      : { draco: false },
  };
}

/**
 * Structural check of a GLB, independent of gltf-transform: header, chunks,
 * JSON, and whether the mesh primitive really is Draco-compressed.
 * @param {Uint8Array} bytes
 */
export function inspectGlb(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const fail = (msg) => { throw new ExportError(`invalid GLB: ${msg}`); };
  if (bytes.byteLength < 20) fail('shorter than a GLB header');
  if (view.getUint32(0, true) !== 0x46546c67) fail('magic is not "glTF"');
  const version = view.getUint32(4, true);
  if (version !== 2) fail(`version ${version}, expected 2`);
  const length = view.getUint32(8, true);
  if (length !== bytes.byteLength) fail(`header length ${length} ≠ ${bytes.byteLength} bytes`);
  const chunks = [];
  for (let o = 12; o < length;) {
    const len = view.getUint32(o, true);
    const type = view.getUint32(o + 4, true);
    if (o + 8 + len > length) fail('chunk overruns the file');
    chunks.push({ type, offset: o + 8, length: len });
    o += 8 + len;
  }
  if (chunks[0]?.type !== 0x4e4f534a) fail('first chunk is not JSON');
  const json = JSON.parse(new TextDecoder().decode(bytes.subarray(chunks[0].offset, chunks[0].offset + chunks[0].length)));
  const bin = chunks.find((c) => c.type === 0x004e4942) ?? null;
  const prims = (json.meshes ?? []).flatMap((m) => m.primitives ?? []);
  const dracoPrims = prims.filter((p) => p.extensions?.[DRACO_EXTENSION]);
  return {
    version, length, json, binLength: bin?.length ?? 0,
    primitives: prims.length,
    draco: {
      used: (json.extensionsUsed ?? []).includes(DRACO_EXTENSION),
      required: (json.extensionsRequired ?? []).includes(DRACO_EXTENSION),
      compressedPrimitives: dracoPrims.length,
      bufferViewBytes: dracoPrims.map((p) => json.bufferViews[p.extensions[DRACO_EXTENSION].bufferView].byteLength),
    },
  };
}

/** File name from the raster, e.g. "ndsm_ABSOLUTE_draco.glb". */
export function exportFileName(raster, draco) {
  const base = String(raster.source || 'terrain').split(/[\\/]/).pop().replace(/\.[^.]*$/, '') || 'terrain';
  const stem = base.endsWith(raster.productType) ? base : `${base}_${raster.productType}`;
  return `${stem}_${raster.mode}${draco ? '_draco' : ''}.glb`.replace(/[^A-Za-z0-9._-]/g, '_');
}

/**
 * Export the raster (+ aligned source image) as a GLB.
 * @param {{raster: import('../data/raster.js').Raster, sourceImage?: import('../data/raster.js').Raster|null,
 *   draco?: boolean, dracoEncoder?: object|null, now?: () => number}} options
 */
export async function exportTerrainGlb({ raster, sourceImage = null, draco = false, dracoEncoder = null, now = () => performance.now() }) {
  const t0 = now();
  if (draco && !dracoEncoder) throw new ExportError('Draco compression requested but no Draco encoder is available; nothing was exported');
  const mesh = buildExportMesh(raster);
  const texture = await buildExportTexture(raster, sourceImage);
  const meta = buildMetadata(raster, mesh, texture, draco);
  const { tile, normals } = mesh;

  const doc = new Document();
  doc.getRoot().getAsset().generator = GENERATOR;
  doc.getRoot().getAsset().extras = { depthwizard: meta };
  const buffer = doc.createBuffer('terrain');
  const acc = (name, type, array) => doc.createAccessor(name).setType(type).setArray(array).setBuffer(buffer);
  const material = doc.createMaterial('terrain').setMetallicFactor(0).setRoughnessFactor(1);
  if (texture.included) {
    const tex = doc.createTexture('source image').setImage(texture.png).setMimeType('image/png');
    material.setBaseColorTexture(tex);
    material.getBaseColorTextureInfo()
      .setWrapS(33071).setWrapT(33071) // CLAMP_TO_EDGE
      .setMagFilter(9729).setMinFilter(9729); // LINEAR
  } else {
    const c = [(BASE_COLOR >> 16) & 255, (BASE_COLOR >> 8) & 255, BASE_COLOR & 255].map((v) => (v / 255) ** 2.2); // sRGB -> linear factor
    material.setBaseColorFactor([...c, 1]);
  }
  const prim = doc.createPrimitive()
    .setAttribute('POSITION', acc('position', 'VEC3', tile.positions))
    .setAttribute('NORMAL', acc('normal', 'VEC3', normals))
    .setAttribute('TEXCOORD_0', acc('uv', 'VEC2', tile.uvs))
    .setIndices(acc('indices', 'SCALAR', tile.indices))
    .setMaterial(material);
  const node = doc.createNode('terrain').setMesh(doc.createMesh('terrain').addPrimitive(prim)).setExtras({ depthwizard: meta });
  doc.createScene('DepthWizard terrain').addChild(node);

  const io = new WebIO();
  if (draco) {
    io.registerExtensions([KHRDracoMeshCompression]).registerDependencies({ 'draco3d.encoder': dracoEncoder });
    doc.createExtension(KHRDracoMeshCompression).setRequired(true).setEncoderOptions({
      method: KHRDracoMeshCompression.EncoderMethod.EDGEBREAKER,
      encodeSpeed: 5,
      decodeSpeed: 5,
      quantizationBits: { ...DRACO_QUANTIZATION },
    });
  }
  const bytes = await io.writeBinary(doc);
  const inspection = inspectGlb(bytes);
  const verified = inspection.draco.used && inspection.draco.compressedPrimitives === inspection.primitives && inspection.primitives > 0;
  if (draco && !verified) throw new ExportError('Draco was requested but the written GLB is not Draco-compressed; refusing to label it Draco');
  if (!draco && inspection.draco.used) throw new ExportError('uncompressed export unexpectedly carries Draco');
  return {
    bytes,
    fileName: exportFileName(raster, draco),
    meta,
    inspection,
    stats: {
      vertices: tile.vertexCount,
      triangles: tile.triangleCount,
      byteLength: bytes.byteLength,
      exportMs: now() - t0,
      draco: { requested: draco, verified: draco && verified },
    },
  };
}

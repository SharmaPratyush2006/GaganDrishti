import path from 'node:path';

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { heightmapFromRaster } from '../src/3d/heightmap.js';
import { buildQuadtree } from '../src/3d/quadtree.js';
import { TerrainLOD } from '../src/3d/terrain.js';
import { describeSourceImage, textureForRaster } from '../src/3d/texture.js';
import { buildTileMesh } from '../src/3d/tileMesh.js';
import { readGeoTiff } from '../src/data/geotiffLoader.js';
import { loadDemo } from '../src/data/load.js';
import { diskFetch, expected, FIXTURES, hasPhase6Outputs, readArrayBuffer } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

const loadImage = (name) => readGeoTiff(readArrayBuffer(path.join(FIXTURES, name)), { source: name, productType: 'image', allBands: true });
const G = expected['tiny_gray_image.tif'];
const heightOnImageGrid = (over = {}) => rasterFrom(grid(G.height, G.width, (r, c) => 10 + r + c),
  { transform: G.transform, crs: G.crs, units: 'metres', ...over });

/** RGBA texel the GPU samples at (u, v): DataTexture with flipY=false stores row 0 at v = 0. */
function texelAt(texture, u, v) {
  const { width, height, data } = texture.image;
  const col = Math.floor(u * width);
  const row = Math.floor(v * height);
  const o = (row * width + col) * 4;
  return Array.from(data.subarray(o, o + 4));
}

describe('source image type is read from the file', () => {
  it('records bands and photometric interpretation for each image', async () => {
    const gray = await loadImage('tiny_gray_image.tif');
    const rgb = await loadImage('tiny_rgb.tif');
    expect([gray.bands, gray.photometric]).toEqual([1, 1]);
    expect([rgb.bands, rgb.photometric]).toEqual([3, 2]);
    expect(rgb.pixels.length).toBe(G.width * G.height * 3);
  });

  it('a 3-band file that is not declared RGB is not treated as RGB', async () => {
    const multi = await loadImage('tiny_multiband.tif');
    const d = describeSourceImage(multi);
    expect(d.kind).toBe('unsupported');
    expect(d.reason).toMatch(/not assumed to be RGB/);
    const t = textureForRaster(heightOnImageGrid(), multi);
    expect(t.texture).toBeNull();
    expect(t.label).not.toMatch(/RGB source image/);
  });
});

describe('texture projection', () => {
  it('1. an aligned image produces a texture', async () => {
    const t = textureForRaster(heightOnImageGrid(), await loadImage('tiny_gray_image.tif'));
    expect(t.alignment.aligned).toBe(true);
    expect(t.texture).toBeInstanceOf(THREE.DataTexture);
    expect([t.texture.image.width, t.texture.image.height]).toEqual([G.width, G.height]);
    expect(t.texture.colorSpace).toBe(THREE.SRGBColorSpace);
  });

  it('2. a grayscale image stays grayscale (R = G = B = the file value; labelled grayscale)', async () => {
    const t = textureForRaster(heightOnImageGrid(), await loadImage('tiny_gray_image.tif'));
    expect(t.status).toBe('grayscale');
    expect(t.label).toBe('Texture: grayscale source image');
    const d = t.texture.image.data;
    for (let r = 0; r < G.height; r++) {
      for (let c = 0; c < G.width; c++) {
        const o = (r * G.width + c) * 4;
        const g = G.pixels[r][c][0];
        expect(Array.from(d.subarray(o, o + 4))).toEqual([g, g, g, 255]);
      }
    }
  });

  it('3. an RGB image keeps its three real channels unchanged', async () => {
    const R = expected['tiny_rgb.tif'];
    const t = textureForRaster(heightOnImageGrid(), await loadImage('tiny_rgb.tif'));
    expect(t.status).toBe('rgb');
    expect(t.label).toBe('Texture: RGB source image');
    const d = t.texture.image.data;
    for (let r = 0; r < R.height; r++) {
      for (let c = 0; c < R.width; c++) {
        const o = (r * R.width + c) * 4;
        expect(Array.from(d.subarray(o, o + 3))).toEqual(R.pixels[r][c]);
      }
    }
  });

  it('4. mismatched dimensions reject projection', async () => {
    const h = rasterFrom(grid(G.height + 1, G.width, () => 1), { transform: G.transform, crs: G.crs, units: 'metres' });
    const t = textureForRaster(h, await loadImage('tiny_gray_image.tif'));
    expect(t.texture).toBeNull();
    expect(t.label).toMatch(/^Texture unavailable — source image is not aligned with the selected raster/);
    expect(t.alignment.reasons.join()).toMatch(/size/);
  });

  it('5. a mismatched transform rejects projection (even by half a pixel)', async () => {
    const shifted = [...G.transform];
    shifted[2] += G.transform[0] / 2;
    const t = textureForRaster(heightOnImageGrid({ transform: shifted }), await loadImage('tiny_gray_image.tif'));
    expect(t.texture).toBeNull();
    expect(t.alignment.reasons.join()).toMatch(/transform/);
  });

  it('6. a mismatched CRS rejects projection', async () => {
    const t = textureForRaster(heightOnImageGrid({ crs: 'EPSG:32644' }), await loadImage('tiny_gray_image.tif'));
    expect(t.texture).toBeNull();
    expect(t.alignment.reasons.join()).toMatch(/CRS/);
  });

  it('7. orientation: every vertex samples the image pixel at its own row and column (no vertical flip)', async () => {
    const img = await loadImage('tiny_gray_image.tif');
    const raster = heightOnImageGrid();
    const { texture } = textureForRaster(raster, img);
    expect(texture.flipY).toBe(false);
    const hm = heightmapFromRaster(raster);
    const tile = buildTileMesh(hm, buildQuadtree(G.width, G.height, { tileSize: 64 }));
    for (let v = 0; v < tile.vertexCount; v++) {
      const p = tile.pixelIndex[v];
      const [row, col] = [Math.floor(p / G.width), p % G.width];
      const [u, vv] = tile.uvs.subarray(2 * v, 2 * v + 2);
      expect(texelAt(texture, u, vv)[0]).toBe(G.pixels[row][col][0]);
    }
    // The fixture's rows differ, so a flipped mapping (v -> 1 - v) would fail above.
    expect(G.pixels[0][0][0]).not.toBe(G.pixels[G.height - 1][0][0]);
    expect(texelAt(texture, 0.5 / G.width, 1 - 0.5 / G.height)[0]).not.toBe(G.pixels[0][0][0]);
  });

  it('8. no fake RGB: a grayscale source never reports or produces distinct colour channels', async () => {
    const t = textureForRaster(heightOnImageGrid(), await loadImage('tiny_gray_image.tif'));
    expect(t.source.kind).toBe('grayscale');
    expect(t.label).not.toMatch(/RGB/);
    const d = t.texture.image.data;
    for (let i = 0; i < d.length; i += 4) expect(d[i] === d[i + 1] && d[i + 1] === d[i + 2]).toBe(true);
  });

  it('9. the texture does not change any height', async () => {
    const raster = heightOnImageGrid();
    const terrain = new TerrainLOD(heightmapFromRaster(raster), { tileSize: 4 });
    terrain.update([0, 50, 50]);
    const before = terrain.group.children.map((m) => Float32Array.from(m.geometry.getAttribute('position').array));
    const geometries = terrain.group.children.map((m) => m.geometry);
    const { texture } = textureForRaster(raster, await loadImage('tiny_gray_image.tif'));
    terrain.setTexture(texture);
    terrain.update([0, 50, 50]);
    expect(terrain.group.children.map((m) => m.geometry)).toEqual(geometries); // same geometry objects
    terrain.group.children.forEach((m, i) => expect(m.geometry.getAttribute('position').array).toEqual(before[i]));
    expect(terrain.group.children[0].material.map).toBe(texture);
    terrain.dispose();
  });

  it('reports a missing source image instead of drawing anything', () => {
    const t = textureForRaster(heightOnImageGrid(), null);
    expect(t.status).toBe('unavailable');
    expect(t.label).toMatch(/no source image is linked/);
  });
});

describe.skipIf(!hasPhase6Outputs())('real Phase 4b source image on the Phase 6 grid', () => {
  it('is grayscale, aligned with the nDSM, and textured as grayscale (no RGB claim)', async () => {
    const d = await loadDemo({}, diskFetch());
    const t = textureForRaster(d.rasters.ndsm, d.rasters.image);
    expect(t.source.kind).toBe('grayscale');
    expect(t.alignment.aligned).toBe(true);
    expect(t.status).toBe('grayscale');
    expect(t.label).not.toMatch(/RGB/);
    // Texel (row, col) = image pixel (row, col), read straight from the decoded file.
    const img = d.rasters.image;
    for (const [r, c] of [[0, 0], [100, 100], [img.height - 1, 3]]) {
      const o = (r * img.width + c) * 4;
      expect(t.texture.image.data[o]).toBe(img.values[r * img.width + c]);
    }
  });
});

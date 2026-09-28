/**
 * Click-to-measure: raycast -> terrain point -> raster pixel -> nDSM value.
 *
 * In plain words: a click shoots a ray from the camera through the mouse
 * position. Where it hits the drawn terrain, we take that spot's horizontal
 * position only, turn it back into a map coordinate, find the raster pixel
 * that contains it (the pixel whose centre the surface sample sits on) and
 * READ THE VALUE FROM THE RASTER. The rendered mesh height is never
 * reported: on a coarse LOD tile the drawn surface is an approximation, but
 * the reported number is the raster's own float32 value at that pixel.
 *
 * Which raster is measured:
 *   Phase 6 data -> the nDSM (height above ground), whatever surface is shown.
 *     The pixel is found through the nDSM's OWN transform, from the clicked
 *     map coordinate, so DSM/DTM/nDSM displays all measure the same nDSM.
 *   Phase 3 .npy -> there is no nDSM; the relative raster itself is measured
 *     and reported as unitless.
 *
 * Pixel convention: pixel (row, col) covers continuous pixel coordinates
 * [col, col+1) x [row, row+1); its sample sits at the centre (col+0.5, row+0.5).
 * A point maps to floor() of its continuous pixel coordinate (as rasterio.rowcol).
 *
 * No valid height (outside the raster, nodata, non-finite, missed the
 * terrain, or a grid that cannot be related) -> ok: false with the reason.
 * No uncertainty is computed or reported here.
 */
import * as THREE from 'three';

import { formatValue, isMetres, PRODUCT_MEANINGS } from '../data/metadata.js';
import { mapToPixel, mapToPixelIndex, valueAt } from '../data/raster.js';
import { worldToMap } from './heightmap.js';

export const NO_VALID_HEIGHT = 'No valid height at this location';

const unitsLabel = (r) => (r.mode === 'ABSOLUTE' && isMetres(r.units) ? 'metres' : 'unitless (relative)');

/**
 * Pixel of `raster` containing the world point, via the displayed heightmap.
 * Shared with the linked cursor (linkedCursor.js).
 * @returns {{row:number,col:number,map:number[]|null}|{error:string}}
 */
export function pixelFor(hm, raster, x, z) {
  const [mx, my] = worldToMap(hm, x, z);
  if (raster.transform) {
    if (!hm.georeferenced) return { error: 'the measured raster is georeferenced but the displayed surface is not' };
    if ((raster.crs ?? null) !== (hm.raster.crs ?? null)) {
      return { error: `CRS differs (displayed ${hm.raster.crs ?? 'none'}, measured ${raster.crs ?? 'none'})` };
    }
    const px = mapToPixelIndex(raster, mx, my);
    return px ? { ...px, map: [mx, my] } : { error: 'outside the raster' };
  }
  // Pixel-space raster: only the displayed grid itself (or an identical one) can be related.
  if (hm.georeferenced || raster.width !== hm.width || raster.height !== hm.height) {
    return { error: 'the measured raster has no georeferencing and is not the displayed grid' };
  }
  const [c, r] = mapToPixel(hm.transform, mx, my);
  const row = Math.floor(r);
  const col = Math.floor(c);
  if (row < 0 || col < 0 || row >= raster.height || col >= raster.width) return { error: 'outside the raster' };
  return { row, col, map: null };
}

/**
 * Measure at a world-space point on the displayed terrain.
 * @param {{x:number,y:number,z:number}} point
 * @param {{heightmap: import('./heightmap.js').Heightmap, measureRaster: import('../data/raster.js').Raster}} ctx
 */
export function measureAtWorldPoint(point, { heightmap: hm, measureRaster }) {
  const base = {
    productType: measureRaster.productType,
    product: PRODUCT_MEANINGS[measureRaster.productType],
    source: measureRaster.source,
    mode: measureRaster.mode,
    units: unitsLabel(measureRaster),
    crs: measureRaster.crs ?? null,
    world: [point.x, point.y, point.z],
  };
  const px = pixelFor(hm, measureRaster, point.x, point.z);
  if (px.error) return { ...base, ok: false, message: NO_VALID_HEIGHT, reason: px.error };
  const { row, col, map } = px;
  const at = { ...base, row, col, map };
  const i = row * measureRaster.width + col;
  if (!measureRaster.valid[i]) return { ...at, ok: false, message: NO_VALID_HEIGHT, reason: 'nodata at this pixel' };
  const value = Number(measureRaster.values[i]);
  if (!Number.isFinite(value)) return { ...at, ok: false, message: NO_VALID_HEIGHT, reason: 'non-finite value at this pixel' };

  // The displayed surface's own value at the same place, when it is a different product.
  let displayed = null;
  if (hm.raster !== measureRaster) {
    const dpx = pixelFor(hm, hm.raster, point.x, point.z);
    const v = dpx.error ? null : valueAt(hm.raster, dpx.row, dpx.col);
    displayed = { productType: hm.raster.productType, product: PRODUCT_MEANINGS[hm.raster.productType], value: v, formatted: formatValue(v, hm.raster, 2) };
  }
  return { ...at, ok: true, value, formatted: formatValue(value, measureRaster, 2), displayed };
}

/**
 * Raycast from canvas client coordinates against the terrain tiles.
 * @returns {{point: THREE.Vector3, tile: object}|null}
 */
export function raycastTerrain({ camera, domElement, terrain, raycaster = new THREE.Raycaster() }, clientX, clientY) {
  const rect = domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -(((clientY - rect.top) / rect.height) * 2 - 1));
  camera.updateMatrixWorld();
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObjects(terrain.group.children, false)[0];
  return hit ? { point: hit.point, tile: { ...hit.object.userData } } : null;
}

/**
 * Measure tool: click handling, one reusable marker, the latest result.
 * A click is a press + release that moved less than CLICK_SLOP_PX, so drags
 * keep orbiting / looking around.
 */
export const CLICK_SLOP_PX = 5;

export function createMeasureTool({ camera, domElement, terrain, scene, measureRaster, render, onMeasure, now = () => performance.now() }) {
  const b = terrain.bounds();
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[2] - b.min[2]) || 1;
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.004 * diag, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xff5a36, depthTest: false }),
  );
  marker.name = 'measure-marker';
  marker.renderOrder = 10;
  marker.visible = false;
  scene.add(marker);
  const raycaster = new THREE.Raycaster();
  let active = false;
  let latest = null;
  let down = null;

  function clickAt(clientX, clientY) {
    const t0 = now();
    const hit = raycastTerrain({ camera, domElement, terrain, raycaster }, clientX, clientY);
    let result;
    if (!measureRaster) {
      result = { ok: false, message: NO_VALID_HEIGHT, reason: 'no raster to measure is loaded' };
    } else if (!hit) {
      result = { ok: false, message: NO_VALID_HEIGHT, reason: 'the click did not hit the terrain', productType: measureRaster.productType };
    } else {
      result = { ...measureAtWorldPoint(hit.point, { heightmap: terrain.heightmap, measureRaster }), tile: hit.tile };
    }
    marker.visible = Boolean(hit);
    if (hit) marker.position.copy(hit.point);
    result.responseMs = now() - t0;
    latest = result;
    render();
    onMeasure?.(result);
    return result;
  }

  const onDown = (e) => {
    if (active && e.button === 0) down = { x: e.clientX, y: e.clientY };
  };
  const onUp = (e) => {
    if (!active || !down || e.button !== 0) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    down = null;
    if (moved < CLICK_SLOP_PX) clickAt(e.clientX, e.clientY);
  };
  domElement.addEventListener('pointerdown', onDown);
  domElement.addEventListener('pointerup', onUp);

  return {
    marker,
    clickAt,
    get active() { return active; },
    get latest() { return latest; },
    setActive(on) {
      active = on;
      if (domElement.style) domElement.style.cursor = on ? 'crosshair' : '';
      if (!on) {
        latest = null;
        marker.visible = false;
        render();
        onMeasure?.(null);
      }
    },
    dispose() {
      domElement.removeEventListener('pointerdown', onDown);
      domElement.removeEventListener('pointerup', onUp);
      scene.remove(marker);
      marker.geometry.dispose();
      marker.material.dispose();
    },
  };
}

/**
 * 2D <-> 3D linked cursor: one raster pixel, shown in a 2D raster panel and
 * as one marker on the 3D terrain.
 *
 * In plain words: the link between the two views is always a raster
 * (row, col) of the displayed surface, never a screen position.
 *
 *   2D panel point -> continuous (col, row) in the panel's raster grid
 *     -> pixel (floor) -> pixel centre (col+0.5, row+0.5)
 *     -> map coordinate (the raster's own affine transform, raster.js)
 *     -> world (x, z) (heightmap.js sampleXZ: minus the recentring origin, z = -map y)
 *   3D terrain hit (x, z) -> map (worldToMap) -> pixel containing it
 *     (measure.js pixelFor: floor, as rasterio.rowcol) -> panel point of its centre
 *
 * The panel draws the raster grid as stored (row 0 at the top, one square
 * cell per pixel). That is the raster's own pixel space, NOT a map: for a
 * rotated, mirrored or non-square-pixel transform the map orientation
 * differs, so the panel shows where map north points (from the transform)
 * instead of assuming north is up. All conversions go through the transform.
 *
 * The panel raster must be on the displayed surface's grid (same size,
 * transform, CRS); otherwise the link is disabled with the reason.
 *
 * The 3D marker is a single reused object (a vertical pin through the whole
 * height range + a head at the pixel's raster value when valid). It is never
 * hit by raycasts and never touches terrain geometry.
 */
import * as THREE from 'three';

import { pixelToMap, mapToPixel, valueAt } from '../data/raster.js';
import { sampleXZ } from './heightmap.js';
import { CLICK_SLOP_PX, pixelFor, raycastTerrain } from './measure.js';
import { textureAlignment } from './texture.js';

export const MARKER_NAME = 'linked-cursor-marker';

/**
 * Fit a W x H raster grid (square cells) into a cw x ch panel, centred.
 * @returns {{scale:number, offX:number, offY:number, width:number, height:number}}
 */
export function panelLayout(width, height, cw, ch) {
  const scale = Math.min(cw / width, ch / height);
  return { scale, offX: (cw - width * scale) / 2, offY: (ch - height * scale) / 2, width, height };
}

/**
 * Panel point -> raster pixel, or null outside the raster.
 * @returns {{row:number, col:number, rowF:number, colF:number}|null}
 */
export function panelToPixel(layout, x, y) {
  const colF = (x - layout.offX) / layout.scale;
  const rowF = (y - layout.offY) / layout.scale;
  if (!(colF >= 0 && rowF >= 0 && colF < layout.width && rowF < layout.height)) return null;
  return { row: Math.floor(rowF), col: Math.floor(colF), rowF, colF };
}

/** Raster pixel -> panel point of its centre. */
export function pixelToPanel(layout, row, col) {
  return [(col + 0.5) * layout.scale + layout.offX, (row + 0.5) * layout.scale + layout.offY];
}

/**
 * Where map north points in the panel's (col, row) space, from the transform.
 * Unit vector in panel coordinates (x right, y down), or null without georeferencing.
 */
export function northInPanel(transform) {
  if (!transform) return null;
  const [c0, r0] = mapToPixel(transform, 0, 0);
  const [c1, r1] = mapToPixel(transform, 0, 1);
  const len = Math.hypot(c1 - c0, r1 - r0);
  return len > 0 ? [(c1 - c0) / len, (r1 - r0) / len] : null;
}

/**
 * The linked location of a displayed-surface pixel.
 * @param {import('./heightmap.js').Heightmap} hm
 * @returns {{row:number,col:number,map:number[]|null,crs:string|null,world:(number|null)[],value:number|null}|null}
 */
export function pixelLocation(hm, row, col) {
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= hm.height || col >= hm.width) return null;
  const [x, z] = sampleXZ(hm, row, col);
  const value = valueAt(hm.raster, row, col);
  return {
    row, col,
    map: hm.georeferenced ? pixelToMap(hm.transform, col + 0.5, row + 0.5) : null,
    crs: hm.georeferenced ? (hm.raster.crs ?? null) : null,
    world: [x, value === null ? null : value * hm.verticalScale, z],
    value,
  };
}

/**
 * World (x, z) on the displayed terrain -> the linked location of the pixel containing it.
 * @returns {ReturnType<typeof pixelLocation>|{error:string}}
 */
export function locationFromWorld(hm, x, z) {
  const px = pixelFor(hm, hm.raster, x, z);
  if (px.error) return { error: px.error };
  return pixelLocation(hm, px.row, px.col);
}

/**
 * Whether a 2D panel raster can be linked to the displayed surface.
 * @returns {{aligned:boolean, reasons:string[]}}
 */
export function linkAlignment(surfaceRaster, panelRaster) {
  if (!panelRaster) return { aligned: false, reasons: ['no 2D raster'] };
  if (panelRaster === surfaceRaster) return { aligned: true, reasons: [] };
  if (!surfaceRaster.transform && !panelRaster.transform && panelRaster.width === surfaceRaster.width
    && panelRaster.height === surfaceRaster.height && !surfaceRaster.crs && !panelRaster.crs) {
    return { aligned: true, reasons: [] }; // same pixel-space grid
  }
  return textureAlignment(surfaceRaster, panelRaster);
}

/**
 * The 3D side: one marker, 3D picks (click, and hover with no button held).
 * @param {{camera: THREE.Camera, domElement: HTMLElement, terrain: import('./terrain.js').TerrainLOD,
 *   scene: THREE.Scene, render: () => void, onLocate?: (loc: object|null) => void,
 *   schedule?: (fn: () => void) => void, now?: () => number}} options
 */
export function createLinkedCursor({ camera, domElement, terrain, scene, render, onLocate,
  schedule = (fn) => requestAnimationFrame(fn), now = () => performance.now() }) {
  const hm = terrain.heightmap;
  const b = terrain.bounds();
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[2] - b.min[2]) || 1;
  const yLow = b.min[1];
  const yHigh = b.max[1] + 0.05 * diag;

  const color = 0x39d5ff;
  const marker = new THREE.Group();
  marker.name = MARKER_NAME;
  const pin = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, yLow, 0), new THREE.Vector3(0, yHigh, 0)]),
    new THREE.LineBasicMaterial({ color, depthTest: false }),
  );
  const head = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.006 * diag),
    new THREE.MeshBasicMaterial({ color, depthTest: false }),
  );
  head.name = 'linked-cursor-head';
  marker.add(pin, head);
  marker.renderOrder = 11;
  pin.renderOrder = 11;
  head.renderOrder = 11;
  marker.traverse((o) => { o.raycast = () => {}; });
  marker.visible = false;
  scene.add(marker);

  const raycaster = new THREE.Raycaster();
  let enabled = false;
  let location = null;
  let down = null;
  let hoverAt = null;
  let hoverPending = false;
  const counters = { picks: 0, hoverPicks: 0, fromPanel: 0 };

  function place(loc) {
    location = loc;
    marker.visible = Boolean(loc);
    if (loc) {
      marker.position.set(loc.world[0], 0, loc.world[2]);
      head.visible = loc.world[1] !== null;
      if (head.visible) head.position.set(0, loc.world[1], 0);
    }
  }

  /** 2D -> 3D: put the cursor on a pixel. */
  function setPixel(row, col) {
    const t0 = now();
    const loc = pixelLocation(hm, row, col);
    place(loc ? { ...loc, from: '2d' } : null);
    render();
    counters.fromPanel++;
    const out = loc ? { ...location, responseMs: now() - t0 } : null;
    onLocate?.(out);
    return out;
  }

  /** 3D -> 2D: pick the terrain under a client point. Misses leave the cursor where it was. */
  function pickAt(clientX, clientY, from = '3d-click') {
    const t0 = now();
    const hit = raycastTerrain({ camera, domElement, terrain, raycaster }, clientX, clientY);
    if (!hit) return { ok: false, reason: 'no terrain under the pointer' };
    const loc = locationFromWorld(hm, hit.point.x, hit.point.z);
    if (loc.error) return { ok: false, reason: loc.error };
    counters.picks++;
    place({ ...loc, from });
    render();
    const out = { ...location, ok: true, hitWorld: [hit.point.x, hit.point.y, hit.point.z], tile: hit.tile, responseMs: now() - t0 };
    onLocate?.(out);
    return out;
  }

  const onDown = (e) => {
    if (enabled && e.button === 0) down = { x: e.clientX, y: e.clientY };
  };
  const onUp = (e) => {
    if (!enabled || !down || e.button !== 0) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    down = null;
    if (moved < CLICK_SLOP_PX) pickAt(e.clientX, e.clientY, '3d-click');
  };
  const onMove = (e) => {
    if (!enabled || e.buttons) return; // dragging navigates; only a free hover picks
    hoverAt = [e.clientX, e.clientY];
    if (hoverPending) return;
    hoverPending = true;
    schedule(() => {
      hoverPending = false;
      if (!enabled || !hoverAt) return;
      counters.hoverPicks++;
      pickAt(hoverAt[0], hoverAt[1], '3d-hover');
    });
  };
  domElement.addEventListener('pointerdown', onDown);
  domElement.addEventListener('pointerup', onUp);
  domElement.addEventListener('pointermove', onMove);

  return {
    marker,
    setPixel,
    pickAt,
    counters,
    get location() { return location; },
    get enabled() { return enabled; },
    setEnabled(on) {
      enabled = on;
      if (!on) {
        hoverAt = null;
        place(null);
        render();
      }
    },
    clear() {
      place(null);
      render();
      onLocate?.(null);
    },
    dispose() {
      domElement.removeEventListener('pointerdown', onDown);
      domElement.removeEventListener('pointerup', onUp);
      domElement.removeEventListener('pointermove', onMove);
      scene.remove(marker);
      pin.geometry.dispose();
      pin.material.dispose();
      head.geometry.dispose();
      head.material.dispose();
    },
  };
}

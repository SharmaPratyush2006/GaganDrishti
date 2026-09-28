/**
 * Step 6: 2D <-> 3D linked cursor. All links go through the raster's own
 * transform; fixtures include rotated, mirrored and non-square-pixel grids.
 */
import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import { createCameraRig } from '../src/3d/cameraRig.js';
import { heightmapFromRaster, sampleXZ } from '../src/3d/heightmap.js';
import {
  createLinkedCursor, linkAlignment, locationFromWorld, MARKER_NAME, northInPanel, panelLayout, panelToPixel, pixelLocation, pixelToPanel,
} from '../src/3d/linkedCursor.js';
import { createMeasureTool, measureAtWorldPoint } from '../src/3d/measure.js';
import { initialCameraPose } from '../src/3d/scene.js';
import { TerrainLOD } from '../src/3d/terrain.js';
import { mapToPixel, pixelToMap } from '../src/data/raster.js';
import { domEvent, fakeCanvas, manualFrames } from './dom.helpers.js';
import { expected } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

const T = expected['tiny_ndsm_cog.tif'].transform;
const ABS = { transform: T, crs: 'EPSG:32643', units: 'metres' };
const pattern = (r, c) => ((r * 7 + c * 13) % 17) + 0.25;
const TRANSFORMS = {
  'north-up': T,
  rotated: [0.3, -0.4, 700000, 0.4, 0.3, 3170000],
  mirrored: [0.5, 0, 700000, 0, 0.5, 3170000], // row increases northwards
  'non-square': [0.5, 0, 700000, 0, -0.25, 3170000],
};

function lookDownAt(x, z, height = 500) {
  const camera = new THREE.PerspectiveCamera(20, 800 / 500, 0.1, 1e5);
  camera.up.set(0, 0, -1);
  camera.position.set(x, height, z);
  camera.lookAt(x, 0, z);
  camera.updateMatrixWorld();
  return camera;
}

describe('2D panel <-> raster pixel', () => {
  it('2D pixel maps to the right raster row/column, and back to the pixel centre', () => {
    const L = panelLayout(20, 12, 300, 300); // 15 px per raster pixel, letterboxed vertically
    expect(L.scale).toBe(15);
    expect([L.offX, L.offY]).toEqual([0, 60]);
    for (const [row, col] of [[0, 0], [5, 7], [11, 19]]) {
      const [x, y] = pixelToPanel(L, row, col);
      expect([x, y]).toEqual([(col + 0.5) * 15, 60 + (row + 0.5) * 15]);
      expect(panelToPixel(L, x, y)).toMatchObject({ row, col });
    }
  });

  it('pixel-centre convention: a panel point stays in its pixel up to the pixel edge', () => {
    const L = panelLayout(20, 12, 300, 300);
    const [cx, cy] = pixelToPanel(L, 4, 6);
    const at = (dx, dy) => { const p = panelToPixel(L, cx + dx * 15, cy + dy * 15); return [p.row, p.col]; };
    expect(at(0.49, 0.49)).toEqual([4, 6]);
    expect(at(-0.49, -0.49)).toEqual([4, 6]);
    expect(at(0.51, 0)).toEqual([4, 7]);
    expect(at(0, -0.51)).toEqual([3, 6]);
  });

  it('points in the letterbox or outside the raster are not linked', () => {
    const L = panelLayout(20, 12, 300, 300);
    expect(panelToPixel(L, 10, 30)).toBeNull();
    expect(panelToPixel(L, 300, 100)).toBeNull();
    expect(panelToPixel(L, -1, 100)).toBeNull();
  });

  it('north is taken from the transform, not assumed up', () => {
    expect(northInPanel(TRANSFORMS['north-up'])).toEqual([0, -1]);
    expect(northInPanel(TRANSFORMS.mirrored)).toEqual([0, 1]);
    const [nx, ny] = northInPanel(TRANSFORMS.rotated);
    expect(nx).toBeCloseTo(0.8, 9); // map +y in pixel space = inverse transform direction
    expect(ny).toBeCloseTo(0.6, 9);
    expect(northInPanel(null)).toBeNull();
  });
});

describe('raster pixel <-> map <-> 3D world', () => {
  for (const [name, t] of Object.entries(TRANSFORMS)) {
    it(`${name} transform: 2D -> 3D -> 2D round trip lands on the same pixel`, () => {
      const r = rasterFrom(grid(9, 11, pattern), { ...ABS, transform: t });
      const hm = heightmapFromRaster(r);
      for (const [row, col] of [[0, 0], [3, 7], [8, 10], [4, 0]]) {
        const loc = pixelLocation(hm, row, col);
        // Map coordinate = the transform applied to the pixel CENTRE.
        const [mx, my] = pixelToMap(t, col + 0.5, row + 0.5);
        expect(loc.map[0]).toBeCloseTo(mx, 9);
        expect(loc.map[1]).toBeCloseTo(my, 9);
        expect(loc.crs).toBe('EPSG:32643');
        // 3D = recentred map, z = -north; y = the raster value.
        expect(loc.world[0]).toBeCloseTo(mx - hm.origin[0], 9);
        expect(loc.world[2]).toBeCloseTo(-(my - hm.origin[1]), 9);
        expect(loc.world[1]).toBe(r.values[row * 11 + col]);
        const back = locationFromWorld(hm, loc.world[0], loc.world[2]);
        expect([back.row, back.col]).toEqual([row, col]);
        // Continuous pixel coordinate of the 3D point is the centre.
        const [c, rr] = mapToPixel(t, loc.world[0] + hm.origin[0], -loc.world[2] + hm.origin[1]);
        expect(c).toBeCloseTo(col + 0.5, 9);
        expect(rr).toBeCloseTo(row + 0.5, 9);
      }
    });
  }

  it('a 3D point anywhere inside a pixel maps to that pixel; past its edge, to the neighbour', () => {
    const t = TRANSFORMS.rotated;
    const hm = heightmapFromRaster(rasterFrom(grid(9, 9, pattern), { ...ABS, transform: t }));
    const worldAt = (colF, rowF) => { const [mx, my] = pixelToMap(t, colF, rowF); return [mx - hm.origin[0], -(my - hm.origin[1])]; };
    const px = (colF, rowF) => { const l = locationFromWorld(hm, ...worldAt(colF, rowF)); return [l.row, l.col]; };
    expect(px(5.01, 3.99)).toEqual([3, 5]);
    expect(px(5.99, 3.01)).toEqual([3, 5]);
    expect(px(6.01, 3.5)).toEqual([3, 6]);
    expect(locationFromWorld(hm, ...worldAt(-0.5, 2)).error).toMatch(/outside the raster/);
  });

  it('nodata: the location is linked but reports no value (no substitute)', () => {
    const r = rasterFrom(grid(5, 5, (row, col) => (row === 2 && col === 3 ? null : 7)), ABS);
    const loc = pixelLocation(heightmapFromRaster(r), 2, 3);
    expect(loc.value).toBeNull();
    expect(loc.world[1]).toBeNull();
    expect([loc.row, loc.col]).toEqual([2, 3]);
  });

  it('pixel-space raster (no CRS/transform): no map coordinate is claimed', () => {
    const r = rasterFrom(grid(4, 6, pattern), { productType: 'relative_height' });
    const hm = heightmapFromRaster(r);
    const loc = pixelLocation(hm, 1, 4);
    expect(loc.map).toBeNull();
    expect(loc.crs).toBeNull();
    const back = locationFromWorld(hm, loc.world[0], loc.world[2]);
    expect([back.row, back.col]).toEqual([1, 4]);
  });

  it('2D and 3D products that are not aligned cannot be linked', () => {
    const a = rasterFrom(grid(4, 6, pattern), ABS);
    const shifted = rasterFrom(grid(4, 6, pattern), { ...ABS, transform: [0.5, 0, 700003, 0, -0.5, 3170000] });
    const other = rasterFrom(grid(5, 6, pattern), ABS);
    expect(linkAlignment(a, a).aligned).toBe(true);
    expect(linkAlignment(a, rasterFrom(grid(4, 6, pattern), ABS)).aligned).toBe(true);
    expect(linkAlignment(a, shifted)).toMatchObject({ aligned: false });
    expect(linkAlignment(a, shifted).reasons.join()).toMatch(/transform/);
    expect(linkAlignment(a, other).reasons.join()).toMatch(/size/);
  });
});

describe('linked cursor tool (3D side)', () => {
  function setup({ transform = T } = {}) {
    const raster = rasterFrom(grid(33, 33, pattern), { ...ABS, transform });
    const hm = heightmapFromRaster(raster);
    const terrain = new TerrainLOD(hm);
    terrain.update([0, 1e4, 0]);
    const [x, z] = sampleXZ(hm, 12, 20);
    const camera = lookDownAt(x, z, 50);
    const scene = new THREE.Scene();
    const el = fakeCanvas();
    const frames = manualFrames();
    const located = [];
    const render = vi.fn();
    const cursor = createLinkedCursor({ camera, domElement: el, terrain, scene, render, onLocate: (l) => located.push(l), schedule: frames.schedule, now: frames.now });
    return { raster, hm, terrain, camera, scene, el, frames, located, render, cursor };
  }

  it('2D -> 3D: setting a pixel moves the single marker to that pixel\'s world position', () => {
    const s = setup({ transform: TRANSFORMS.rotated });
    const loc = s.cursor.setPixel(7, 9);
    const [x, z] = sampleXZ(s.hm, 7, 9);
    expect(s.cursor.marker.visible).toBe(true);
    expect(s.cursor.marker.position.x).toBeCloseTo(x, 9);
    expect(s.cursor.marker.position.z).toBeCloseTo(z, 9);
    expect(s.cursor.marker.children[1].position.y).toBe(s.raster.values[7 * 33 + 9]);
    expect(s.located.at(-1)).toMatchObject({ row: 7, col: 9, from: '2d' });
    expect(loc.responseMs).toBeGreaterThanOrEqual(0);
    expect(s.render).toHaveBeenCalled();
  });

  it('3D -> 2D: a terrain click reports the same pixel as click-to-measure', () => {
    const s = setup();
    s.cursor.setEnabled(true);
    s.el.dispatchEvent(domEvent('pointerdown', { button: 0, clientX: 400, clientY: 250 }));
    s.el.dispatchEvent(domEvent('pointerup', { button: 0, clientX: 400, clientY: 250 }));
    const loc = s.located.at(-1);
    expect([loc.row, loc.col]).toEqual([12, 20]);
    expect(loc.from).toBe('3d-click');
    const m = measureAtWorldPoint({ x: loc.hitWorld[0], y: 0, z: loc.hitWorld[2] }, { heightmap: s.hm, measureRaster: s.raster });
    expect([m.row, m.col]).toEqual([loc.row, loc.col]);
    expect(loc.value).toBe(m.value);
  });

  it('3D hover picks at most once per animation frame and only with no button held', () => {
    const s = setup();
    s.cursor.setEnabled(true);
    for (let i = 0; i < 5; i++) s.el.dispatchEvent(domEvent('pointermove', { buttons: 0, clientX: 400 + i, clientY: 250 }));
    expect(s.frames.queue.length).toBe(1);
    s.frames.runFrame();
    expect(s.cursor.counters.hoverPicks).toBe(1);
    expect(s.located.at(-1).from).toBe('3d-hover');
    s.el.dispatchEvent(domEvent('pointermove', { buttons: 1, clientX: 300, clientY: 250 })); // dragging = navigating
    expect(s.frames.queue.length).toBe(0);
  });

  it('a drag does not pick; a disabled cursor does nothing and hides the marker', () => {
    const s = setup();
    s.cursor.setEnabled(true);
    s.el.dispatchEvent(domEvent('pointerdown', { button: 0, clientX: 400, clientY: 250 }));
    s.el.dispatchEvent(domEvent('pointerup', { button: 0, clientX: 440, clientY: 250 }));
    expect(s.located.length).toBe(0);
    s.cursor.setPixel(3, 3);
    s.cursor.setEnabled(false);
    expect(s.cursor.marker.visible).toBe(false);
    s.el.dispatchEvent(domEvent('pointerdown', { button: 0, clientX: 400, clientY: 250 }));
    s.el.dispatchEvent(domEvent('pointerup', { button: 0, clientX: 400, clientY: 250 }));
    s.el.dispatchEvent(domEvent('pointermove', { buttons: 0, clientX: 400, clientY: 250 }));
    expect(s.frames.queue.length).toBe(0);
    expect(s.located.length).toBe(1);
  });

  it('the marker is reused (one object after many moves) and is never hit by raycasts', () => {
    const s = setup();
    for (let i = 0; i < 50; i++) s.cursor.setPixel(i % 33, (i * 7) % 33);
    expect(s.scene.children.filter((o) => o.name === MARKER_NAME).length).toBe(1);
    expect(s.scene.children.length).toBe(1);
    const ray = new THREE.Raycaster(new THREE.Vector3(s.cursor.marker.position.x, 1e3, s.cursor.marker.position.z), new THREE.Vector3(0, -1, 0));
    expect(ray.intersectObject(s.cursor.marker, true)).toEqual([]);
    expect(s.terrain.group.children.includes(s.cursor.marker)).toBe(false);
  });

  it('nodata pixel: marker pin shown, head hidden (no invented height)', () => {
    const raster = rasterFrom(grid(9, 9, (r, c) => (r === 4 && c === 4 ? null : 2)), ABS);
    const terrain = new TerrainLOD(heightmapFromRaster(raster));
    terrain.update([0, 1e4, 0]);
    const cursor = createLinkedCursor({ camera: lookDownAt(0, 0), domElement: fakeCanvas(), terrain, scene: new THREE.Scene(), render() {} });
    const loc = cursor.setPixel(4, 4);
    expect(loc.value).toBeNull();
    expect(cursor.marker.visible).toBe(true);
    expect(cursor.marker.children[1].visible).toBe(false);
  });

  it('dispose removes the marker, frees its resources and detaches listeners', () => {
    const s = setup();
    const disposed = [];
    s.cursor.marker.traverse((o) => {
      o.geometry?.addEventListener('dispose', () => disposed.push('geometry'));
      o.material?.addEventListener('dispose', () => disposed.push('material'));
    });
    s.cursor.setEnabled(true);
    s.cursor.dispose();
    expect(s.scene.children.length).toBe(0);
    expect(disposed.sort()).toEqual(['geometry', 'geometry', 'material', 'material']);
    s.el.dispatchEvent(domEvent('pointerdown', { button: 0, clientX: 400, clientY: 250 }));
    s.el.dispatchEvent(domEvent('pointerup', { button: 0, clientX: 400, clientY: 250 }));
    expect(s.located.length).toBe(0);
  });
});

describe('linked cursor alongside navigation and click-to-measure', () => {
  it('one click measures AND links the same pixel; each tool keeps its own single marker', () => {
    const raster = rasterFrom(grid(33, 33, pattern), ABS);
    const hm = heightmapFromRaster(raster);
    const terrain = new TerrainLOD(hm);
    terrain.update([0, 1e4, 0]);
    const [x, z] = sampleXZ(hm, 12, 20);
    const camera = lookDownAt(x, z, 50);
    const scene = new THREE.Scene();
    const el = fakeCanvas();
    const measured = [];
    const linked = [];
    const measure = createMeasureTool({ camera, domElement: el, terrain, scene, measureRaster: raster, render() {}, onMeasure: (r) => measured.push(r) });
    const cursor = createLinkedCursor({ camera, domElement: el, terrain, scene, render() {}, onLocate: (l) => linked.push(l), schedule: () => {} });
    measure.setActive(true);
    cursor.setEnabled(true);
    el.dispatchEvent(domEvent('pointerdown', { button: 0, clientX: 400, clientY: 250 }));
    el.dispatchEvent(domEvent('pointerup', { button: 0, clientX: 400, clientY: 250 }));
    expect(measured.length).toBe(1);
    expect(linked.length).toBe(1);
    expect([measured[0].row, measured[0].col]).toEqual([linked[0].row, linked[0].col]);
    expect(measured[0].value).toBe(linked[0].value);
    expect(scene.children.map((o) => o.name).sort()).toEqual([MARKER_NAME, 'measure-marker'].sort());
    // Measurement ignores the cursor marker: a second click measures the terrain, not the pin.
    cursor.setPixel(12, 20);
    const again = measure.clickAt(400, 250);
    expect([again.row, again.col]).toEqual([12, 20]);
    measure.setActive(false);
    expect(cursor.marker.visible).toBe(true); // exiting measure mode does not clear the cursor
  });

  it('orbit and first-person keep working with the cursor enabled; the cursor never changes the camera mode', () => {
    const raster = rasterFrom(grid(65, 65, (r, c) => (r > 20 && r < 30 && c > 20 && c < 30 ? 20 : 1)), ABS);
    const terrain = new TerrainLOD(heightmapFromRaster(raster), { tileSize: 8 });
    const camera = new THREE.PerspectiveCamera(50, 1.6, 0.1, 1e4);
    const el = fakeCanvas();
    const frames = manualFrames();
    const rig = createCameraRig({ camera, domElement: el, terrain, render() {}, schedule: frames.schedule, now: frames.now });
    const pose = initialCameraPose(terrain.bounds());
    rig.reset(pose);
    const cursor = createLinkedCursor({ camera, domElement: el, terrain, scene: new THREE.Scene(), render() {}, schedule: frames.schedule, now: frames.now });
    cursor.setEnabled(true);
    const lodBefore = terrain.lastStats.selectionKey ?? terrain.selectionKey;

    // Orbit zoom still re-selects LOD.
    rig.controls.dollyIn(4);
    rig.controls.update();
    frames.flush();
    expect(terrain.selectionKey).not.toBe(lodBefore);
    expect(rig.mode).toBe('orbit');

    // Hover + click in 3D and 2D picks leave the camera untouched.
    const camPos = camera.position.toArray();
    el.dispatchEvent(domEvent('pointermove', { buttons: 0, clientX: 400, clientY: 250 }));
    frames.flush();
    cursor.setPixel(10, 10);
    expect(camera.position.toArray()).toEqual(camPos);
    expect(rig.mode).toBe('orbit');

    // First person still switches and moves.
    rig.setMode('firstPerson');
    expect(rig.mode).toBe('firstPerson');
    cursor.setPixel(40, 40);
    expect(rig.mode).toBe('firstPerson');
    rig.setMode('orbit');
    expect(rig.mode).toBe('orbit');
    rig.dispose();
    cursor.dispose();
  });
});

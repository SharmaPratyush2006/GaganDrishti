import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import { createCameraRig, MAX_POLAR_DEG } from '../src/3d/cameraRig.js';
import { heightmapFromRaster, sampleXZ } from '../src/3d/heightmap.js';
import { initialCameraPose } from '../src/3d/scene.js';
import { TerrainLOD } from '../src/3d/terrain.js';
import { expected } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

/** The DOM surface OrbitControls uses, built on Node's EventTarget (no jsdom). */
function fakeCanvas() {
  const el = new EventTarget();
  const doc = new EventTarget();
  Object.assign(el, {
    style: {}, ownerDocument: doc, getRootNode: () => doc, clientWidth: 800, clientHeight: 500,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 500 }),
    setPointerCapture() {}, releasePointerCapture() {},
  });
  return el;
}

function setup() {
  const raster = rasterFrom(grid(129, 129, (r, c) => (r > 40 && r < 60 && c > 40 && c < 60 ? 30 : 1)),
    { transform: expected['tiny_ndsm_cog.tif'].transform, crs: 'EPSG:32643', units: 'metres' });
  const hm = heightmapFromRaster(raster);
  const terrain = new TerrainLOD(hm, { tileSize: 8 });
  const camera = new THREE.PerspectiveCamera(50, 1.6, 0.1, 10000);
  const queue = [];
  const render = vi.fn();
  const updateSpy = vi.spyOn(terrain, 'update');
  const rig = createCameraRig({ camera, domElement: fakeCanvas(), terrain, render, schedule: (fn) => queue.push(fn) });
  const pose = initialCameraPose(terrain.bounds());
  rig.reset(pose);
  const flush = () => { while (queue.length) queue.shift()(); };
  return { hm, terrain, camera, rig, pose, render, updateSpy, flush, queue };
}

describe('orbit controls on the existing camera and LOD', () => {
  it('attaches OrbitControls to the viewer camera', () => {
    const { rig, camera } = setup();
    expect(rig.controls.object).toBe(camera);
    expect(rig.controls.constructor.name).toBe('OrbitControls');
  });

  it('starts from the overview with the target inside the terrain bounds', () => {
    const { rig, terrain, pose } = setup();
    const b = terrain.bounds();
    const t = rig.status().target;
    for (const k of [0, 1, 2]) {
      expect(t[k]).toBeGreaterThanOrEqual(b.min[k]);
      expect(t[k]).toBeLessThanOrEqual(b.max[k]);
    }
    expect(rig.status().position).toEqual(pose.position);
  });

  it('limits come from the terrain and allow close building inspection', () => {
    const { rig, terrain } = setup();
    const b = terrain.bounds();
    const diag = Math.hypot(b.max[0] - b.min[0], b.max[2] - b.min[2]);
    expect(rig.limits.minDistance).toBeLessThan(0.01 * diag);
    expect(rig.limits.maxDistance).toBeGreaterThan(rig.status().distance);
    expect(rig.limits.maxPolarDeg).toBe(MAX_POLAR_DEG);
  });

  it('camera movement updates the SAME TerrainLOD, once per frame, and changes the LOD', () => {
    const { rig, terrain, hm, updateSpy, flush, queue } = setup();
    const before = terrain.lastStats;
    updateSpy.mockClear();
    // Move close to a corner through the controls themselves.
    const [x, z] = sampleXZ(hm, 5, 5);
    rig.controls.target.set(x, 1, z);
    rig.controls.object.position.set(x + 3, 6, z + 3);
    rig.controls.update(); // emits 'change'
    rig.controls.update(); // second change in the same frame
    expect(queue.length).toBe(1); // coalesced
    flush();
    expect(updateSpy).toHaveBeenCalledTimes(1);
    const after = terrain.lastStats;
    expect(after.selectionChanged).toBe(true);
    expect(Object.keys(after.tilesByLevel)).not.toEqual(Object.keys(before.tilesByLevel));
    expect(Object.keys(after.tilesByLevel).some((k) => k.includes('stride 1'))).toBe(true);
  });

  it('returning to a previous view re-uses cached tiles (no rebuild)', () => {
    const { rig, terrain, hm, pose, flush } = setup();
    const [x, z] = sampleXZ(hm, 5, 5);
    rig.controls.target.set(x, 1, z);
    rig.controls.object.position.set(x + 3, 6, z + 3);
    rig.controls.update();
    flush();
    const cached = terrain.cache.size;
    const back = rig.reset(pose);
    expect(back.tilesBuiltThisUpdate).toBe(0);
    expect(terrain.cache.size).toBe(cached);
  });

  it('a camera move that keeps the same selection swaps no meshes', () => {
    const { rig, terrain, flush } = setup();
    const children = [...terrain.group.children];
    rig.controls.object.position.x += 0.01;
    rig.controls.update();
    flush();
    expect(terrain.lastStats.selectionChanged).toBe(false);
    expect(terrain.group.children).toEqual(children);
  });

  it('panning cannot move the target off the terrain', () => {
    const { rig, terrain, flush } = setup();
    rig.controls.target.set(1e6, 0, -1e6);
    rig.controls.object.position.set(1e6, 100, -1e6 + 100);
    rig.controls.update();
    flush();
    const b = terrain.bounds();
    const t = rig.status().target;
    expect(t[0]).toBe(b.max[0]);
    expect(t[2]).toBe(b.min[2]);
  });
});

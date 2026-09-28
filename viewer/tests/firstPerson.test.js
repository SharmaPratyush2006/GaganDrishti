import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import { createCameraRig } from '../src/3d/cameraRig.js';
import { EYE_HEIGHT_FRACTION, FAST_MULTIPLIER, groundHeightAt, MAX_PITCH_DEG, SPEED_FRACTION, terrainScale } from '../src/3d/firstPerson.js';
import { heightmapFromRaster, sampleXZ } from '../src/3d/heightmap.js';
import { initialCameraPose } from '../src/3d/scene.js';
import { TerrainLOD } from '../src/3d/terrain.js';
import { readNpyRaster } from '../src/data/npy.js';
import { domEvent, fakeCanvas, manualFrames } from './dom.helpers.js';
import { expected } from './helpers.js';
import { grid, rasterFrom } from './mesh3d.helpers.js';

const T = expected['tiny_ndsm_cog.tif'].transform;
const inBuilding = (r, c) => r > 40 && r < 60 && c > 40 && c < 60;
const BUILDING_H = 30;

function setup({ transform = T, raster } = {}) {
  const r = raster ?? rasterFrom(grid(129, 129, (row, col) => (inBuilding(row, col) ? BUILDING_H : 1)),
    { transform, crs: 'EPSG:32643', units: 'metres' });
  const hm = heightmapFromRaster(r);
  const terrain = new TerrainLOD(hm, { tileSize: 8 });
  const camera = new THREE.PerspectiveCamera(50, 1.6, 0.01, 1e5);
  const frames = manualFrames();
  const el = fakeCanvas();
  const updateSpy = vi.spyOn(terrain, 'update');
  const rig = createCameraRig({ camera, domElement: el, terrain, render: () => {}, schedule: frames.schedule, now: frames.now });
  const pose = initialCameraPose(terrain.bounds());
  rig.reset(pose);
  return { r, hm, terrain, camera, frames, el, rig, pose, updateSpy, fp: rig.firstPerson };
}

/** Hold a key for `ms` of simulated time through the real listeners. */
function hold(s, code, ms) {
  s.el.ownerDocument.dispatchEvent(domEvent('keydown', { code }));
  const steps = Math.round(ms / 20);
  for (let i = 0; i < steps; i++) {
    s.frames.advance(20);
    s.frames.runFrame();
  }
  s.el.ownerDocument.dispatchEvent(domEvent('keyup', { code }));
  s.frames.runFrame();
  s.frames.runFrame();
}

describe('first-person mode', () => {
  it('1. drives the existing camera (no second camera)', () => {
    const s = setup();
    s.rig.setMode('firstPerson');
    expect(s.rig.controls.object).toBe(s.camera);
    const before = s.camera.position.clone();
    s.fp.move(1, 0, 0);
    expect(s.camera.position.equals(before)).toBe(false);
  });

  it('2-3. orbit -> first person: stands at the orbit target at eye height, facing the orbit view direction', () => {
    const s = setup();
    expect(s.rig.mode).toBe('orbit');
    s.rig.setMode('firstPerson');
    expect(s.rig.mode).toBe('firstPerson');
    expect(s.rig.controls.enabled).toBe(false);
    const { eyeHeight } = terrainScale(s.terrain);
    const ground = groundHeightAt(s.hm, s.pose.target[0], s.pose.target[2], terrainScale(s.terrain).collisionRadius);
    expect(s.camera.position.x).toBeCloseTo(s.pose.target[0], 9);
    expect(s.camera.position.z).toBeCloseTo(s.pose.target[2], 9);
    expect(s.camera.position.y).toBeCloseTo(ground + eyeHeight, 9);
    // The overview looks north (-z); first person keeps that heading.
    const dir = new THREE.Vector3();
    s.camera.getWorldDirection(dir);
    expect(dir.z).toBeLessThan(-0.99);
  });

  it('4. first person -> orbit: orbit resumes around a terrain point ahead, controls enabled and working', () => {
    const s = setup();
    s.rig.setMode('firstPerson');
    s.rig.setMode('orbit');
    expect(s.rig.controls.enabled).toBe(true);
    const st = s.rig.status();
    const b = s.terrain.bounds();
    for (const k of [0, 2]) expect(st.target[k]).toBeGreaterThanOrEqual(b.min[k]);
    expect(st.target[2]).toBeLessThan(st.position[2]); // ahead = north of the camera
    expect(st.distance).toBeGreaterThanOrEqual(s.rig.limits.minDistance);
    expect(st.distance).toBeLessThanOrEqual(s.rig.limits.maxDistance);
    // Orbit input still drives the LOD.
    s.updateSpy.mockClear();
    s.rig.controls.object.position.y += 5;
    s.rig.controls.update();
    s.frames.flush();
    expect(s.updateSpy).toHaveBeenCalledTimes(1);
  });

  it('5-6. W moves forward and D strafes right by speed x time, through the real key listeners', () => {
    const s = setup();
    s.rig.setMode('firstPerson');
    const { speed } = terrainScale(s.terrain);
    const p0 = s.camera.position.clone();
    hold(s, 'KeyW', 200); // facing north (-z)
    expect(s.camera.position.z - p0.z).toBeCloseTo(-speed * 0.2, 6);
    expect(s.camera.position.x).toBeCloseTo(p0.x, 9);
    const p1 = s.camera.position.clone();
    hold(s, 'KeyD', 100);
    expect(s.camera.position.x - p1.x).toBeCloseTo(speed * 0.1, 6);
    const p2 = s.camera.position.clone();
    s.el.ownerDocument.dispatchEvent(domEvent('keydown', { code: 'ShiftLeft' }));
    hold(s, 'KeyS', 100);
    expect(s.camera.position.z - p2.z).toBeCloseTo(speed * FAST_MULTIPLIER * 0.1, 6);
  });

  it('mouse drag looks around, pitch is limited', () => {
    const s = setup();
    s.rig.setMode('firstPerson');
    s.el.dispatchEvent(domEvent('pointerdown', { button: 0, clientX: 100, clientY: 100 }));
    s.el.ownerDocument.dispatchEvent(domEvent('pointermove', { clientX: 300, clientY: 100 }));
    s.el.ownerDocument.dispatchEvent(domEvent('pointerup', {}));
    expect(s.fp.status().yawDeg).toBeLessThan(-10); // dragged right -> turned right
    s.fp.lookBy(0, -1e6);
    expect(s.fp.status().pitchDeg).toBeCloseTo(MAX_PITCH_DEG, 9);
  });

  it('keys do nothing in orbit mode', () => {
    const s = setup();
    const p0 = s.camera.position.clone();
    hold(s, 'KeyW', 200);
    expect(s.camera.position.equals(p0)).toBe(true);
  });

  it('7. stays inside the terrain bounds', () => {
    const s = setup();
    s.rig.setMode('firstPerson');
    s.fp.move(1e7, 1e7, 1e7);
    const b = s.terrain.bounds();
    expect(s.camera.position.z).toBe(b.min[2]);
    expect(s.camera.position.x).toBe(b.max[0]);
    expect(s.camera.position.y).toBeLessThanOrEqual(b.max[1] + terrainScale(s.terrain).diag);
  });

  it('8. never goes below the surface, and rises over a building instead of passing through it', () => {
    const s = setup();
    s.rig.setMode('firstPerson');
    const { eyeHeight } = terrainScale(s.terrain);
    s.fp.move(0, 0, -1e7);
    expect(s.camera.position.y).toBeCloseTo(1 + eyeHeight, 9); // ground value 1 from the raster
    // Walk onto the building (rows/cols 41..59): read its height from the raster, not a constant.
    const [bx, bz] = sampleXZ(s.hm, 50, 50);
    s.camera.position.set(bx, 0, bz);
    s.fp.move(0, 0, 0);
    const roof = s.r.values[50 * 129 + 50];
    expect(s.camera.position.y).toBeCloseTo(roof + eyeHeight, 9);
  });

  it('9. scale follows the terrain extent, not a fixture constant', () => {
    const small = setup();
    const big = setup({ transform: [2, 0, 700000, 0, -2, 3170000] });
    const a = terrainScale(small.terrain);
    const b = terrainScale(big.terrain);
    expect(b.eyeHeight / a.eyeHeight).toBeCloseTo(4, 9);
    expect(b.speed / a.speed).toBeCloseTo(4, 9);
    expect(a.eyeHeight).toBeCloseTo(EYE_HEIGHT_FRACTION * a.diag, 12);
    expect(a.speed).toBeCloseTo(SPEED_FRACTION * a.diag, 12);
  });

  it('10. reset view restores the starting state in each mode', () => {
    const s = setup();
    s.rig.setMode('firstPerson');
    const start = s.camera.position.clone();
    s.fp.move(10, 5, 3);
    s.fp.lookBy(200, 50);
    s.rig.reset(s.pose);
    expect(s.camera.position.distanceTo(start)).toBeLessThan(1e-9);
    expect(s.fp.status().yawDeg).toBeCloseTo(0, 9);
    expect(s.fp.status().pitchDeg).toBe(0);
    s.rig.setMode('orbit');
    s.rig.reset(s.pose);
    expect(s.rig.status().position).toEqual(s.pose.position);
    expect(s.rig.status().target).toEqual(s.pose.target);
  });

  it('11-12. movement updates the existing LOD (finer near the eye) and re-uses cached tiles', () => {
    const s = setup();
    const overview = s.terrain.lastStats;
    s.updateSpy.mockClear();
    s.rig.setMode('firstPerson');
    expect(s.updateSpy).toHaveBeenCalled();
    const near = s.terrain.lastStats;
    expect(Object.keys(near.tilesByLevel).some((k) => k.includes('stride 1'))).toBe(true);
    expect(near.triangles).toBeGreaterThan(overview.triangles);
    const calls = s.updateSpy.mock.calls.length;
    hold(s, 'KeyW', 400);
    expect(s.updateSpy.mock.calls.length).toBeGreaterThan(calls);
    s.rig.reset(s.pose); // back to the first-person start: already built
    expect(s.terrain.lastStats.tilesBuiltThisUpdate).toBe(0);
    s.rig.setMode('orbit');
    s.rig.reset(s.pose);
    expect(s.terrain.lastStats.tilesBuiltThisUpdate).toBe(0);
  });

  it('works on a RELATIVE .npy (pixel-space, display scale) without changing its values', () => {
    const vals = new Float32Array(33 * 33).map((_, i) => (i % 33 > 10 && i % 33 < 20 ? 2.5 : 0.1));
    const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (33, 33), }";
    const padded = header + ' '.repeat(64 - ((10 + header.length + 1) % 64)) + '\n';
    const bytes = new Uint8Array(10 + padded.length + vals.byteLength);
    bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
    new DataView(bytes.buffer).setUint16(8, padded.length, true);
    bytes.set(new TextEncoder().encode(padded), 10);
    bytes.set(new Uint8Array(vals.buffer), 10 + padded.length);
    const raster = readNpyRaster(bytes.buffer);
    const copy = Float32Array.from(raster.values);
    const s = setup({ raster });
    expect(s.hm.raster.mode).toBe('RELATIVE');
    s.rig.setMode('firstPerson');
    s.fp.move(0, 0, -1e6);
    const ground = groundHeightAt(s.hm, s.camera.position.x, s.camera.position.z, terrainScale(s.terrain).collisionRadius);
    expect(s.camera.position.y).toBeCloseTo(ground + terrainScale(s.terrain).eyeHeight, 9);
    expect(raster.values).toEqual(copy);
  });
});

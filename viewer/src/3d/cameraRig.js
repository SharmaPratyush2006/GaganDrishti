/**
 * Camera navigation wired to the existing camera and the existing TerrainLOD.
 *
 * In plain words: one camera, two ways to drive it.
 *   ORBIT         three.js OrbitControls (left-drag rotate, right-drag pan,
 *                 wheel/pinch zoom) around a target on the terrain.
 *   FIRST PERSON  firstPerson.js (WASD/QE + drag to look), standing on the terrain.
 * Whenever the camera moves -- in either mode, at most once per animation
 * frame -- the SAME TerrainLOD re-selects tiles for the new camera position
 * (re-using its tile cache) and the scene is re-rendered. There is no second
 * LOD system and no full rebuild on input events.
 *
 * Orbit limits (derived from the terrain's own bounding box, not hard-coded):
 *   minDistance = 0.5% of the terrain's horizontal diagonal -- close enough
 *                 to inspect a single building
 *   maxDistance = 3 x the diagonal -- the whole terrain stays in view
 *   maxPolarAngle = 85 deg -- the camera cannot go below the horizon
 *   the orbit target is kept inside the terrain's bounding box, so panning
 *   cannot lose the scene.
 *
 * Switching modes:
 *   orbit -> first person: stand at the orbit target, facing the way the
 *     orbit camera looked (horizontally).
 *   first person -> orbit: keep the camera where it is and orbit a point on
 *     the terrain ahead of it.
 * Reset view: orbit -> the initial overview; first person -> standing at the
 *   overview's target, facing the same way as the overview.
 */
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { createFirstPerson } from './firstPerson.js';

export const MIN_DISTANCE_FRACTION = 0.005;
export const MAX_DISTANCE_FACTOR = 3;
export const MAX_POLAR_DEG = 85;
/** Distance of the orbit target ahead of a first-person camera, in eye heights. */
const HANDBACK_EYE_HEIGHTS = 20;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * @param {{
 *   camera: import('three').PerspectiveCamera,
 *   domElement: HTMLElement,
 *   terrain: import('./terrain.js').TerrainLOD,
 *   render: () => void,
 *   onUpdate?: (info: {lod: object, camera: object}) => void,
 *   schedule?: (fn: () => void) => void,
 *   now?: () => number,
 *   Controls?: typeof OrbitControls,
 * }} options
 */
export function createCameraRig({ camera, domElement, terrain, render, onUpdate, schedule = (fn) => requestAnimationFrame(fn), now, Controls = OrbitControls }) {
  const controls = new Controls(camera, domElement);
  const b = terrain.bounds();
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[2] - b.min[2]) || 1;
  controls.minDistance = MIN_DISTANCE_FRACTION * diag;
  controls.maxDistance = MAX_DISTANCE_FACTOR * diag;
  controls.maxPolarAngle = (MAX_POLAR_DEG * Math.PI) / 180;
  controls.enableDamping = false; // render on demand: every change is final

  const limits = {
    minDistance: controls.minDistance,
    maxDistance: controls.maxDistance,
    maxPolarDeg: MAX_POLAR_DEG,
    targetBox: { min: [...b.min], max: [...b.max] },
  };

  const counters = { frames: 0, changeEvents: 0, modeSwitches: 0 };
  let mode = 'orbit';
  let pending = false;
  let resetting = false; // programmatic moves render themselves; their 'change' must not queue another frame

  function requestFrame() {
    if (!pending && !resetting) {
      pending = true;
      schedule(frame);
    }
  }
  const fp = createFirstPerson({ camera, domElement, terrain, requestFrame, schedule, ...(now ? { now } : {}) });

  /** Keep the orbit target over the terrain; move the camera with it so the view does not jump. */
  function clampTarget() {
    const t = controls.target;
    const nx = clamp(t.x, b.min[0], b.max[0]);
    const ny = clamp(t.y, b.min[1], b.max[1]);
    const nz = clamp(t.z, b.min[2], b.max[2]);
    if (nx !== t.x || ny !== t.y || nz !== t.z) {
      camera.position.x += nx - t.x;
      camera.position.y += ny - t.y;
      camera.position.z += nz - t.z;
      t.set(nx, ny, nz);
      camera.lookAt(t);
    }
  }

  function status() {
    const offset = camera.position.clone().sub(controls.target);
    return {
      mode,
      position: camera.position.toArray(),
      target: controls.target.toArray(),
      distance: offset.length(),
      polarDeg: (controls.getPolarAngle() * 180) / Math.PI,
      azimuthDeg: (controls.getAzimuthalAngle() * 180) / Math.PI,
      limits,
      firstPerson: fp.status(),
      frames: counters.frames,
      changeEvents: counters.changeEvents,
      modeSwitches: counters.modeSwitches,
    };
  }

  function frame() {
    pending = false;
    counters.frames++;
    if (mode === 'orbit') clampTarget();
    const lod = terrain.update(camera.position.toArray());
    render();
    onUpdate?.({ lod, camera: status() });
    return lod;
  }
  const onChange = () => {
    counters.changeEvents++;
    requestFrame();
  };
  controls.addEventListener('change', onChange);

  function orbitAt(target) {
    controls.target.set(...target);
    resetting = true;
    controls.update();
    resetting = false;
  }

  return {
    controls,
    firstPerson: fp,
    limits,
    status,
    frame,
    get mode() { return mode; },
    /**
     * Switch navigation mode ('orbit' | 'firstPerson'); returns the LOD stats
     * of the immediate update.
     */
    setMode(next) {
      if (next !== 'orbit' && next !== 'firstPerson') throw new Error(`unknown camera mode ${next}`);
      if (next === mode) return terrain.lastStats;
      counters.modeSwitches++;
      if (next === 'firstPerson') {
        controls.enabled = false;
        fp.enable(fp.poseFromOrbit(camera.position.toArray(), controls.target.toArray()));
      } else {
        const target = fp.pointAhead(HANDBACK_EYE_HEIGHTS * fp.scale.eyeHeight);
        fp.disable();
        controls.enabled = true;
        orbitAt(target);
      }
      mode = next;
      pending = false;
      return frame();
    },
    /** Return to the starting view of the current mode (see header). */
    reset({ position, target }) {
      pending = false;
      if (mode === 'firstPerson') {
        fp.enable(fp.poseFromOrbit(position, target));
      } else {
        camera.position.set(...position);
        orbitAt(target);
      }
      return frame();
    },
    dispose() {
      controls.removeEventListener('change', onChange);
      controls.dispose();
      fp.dispose();
    },
  };
}

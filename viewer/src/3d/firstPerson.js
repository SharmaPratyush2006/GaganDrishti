/**
 * First-person flythrough on the existing camera.
 *
 * In plain words: walk/fly over the terrain like a person with a camera.
 *   W / S       forward / backward (along the view direction, kept horizontal)
 *   A / D       strafe left / right
 *   Q / E       down / up
 *   Shift       faster
 *   drag mouse  look around (no pointer lock: works everywhere, including
 *               automated browsers)
 *
 * Scale comes from the terrain itself, never from a particular fixture:
 *   eye height = 0.5% of the terrain's horizontal diagonal
 *   speed      = 4% of the diagonal per second (x4 with Shift)
 *
 * Staying on / above the terrain, using the loaded raster's own values:
 *   the camera may never be lower than (highest valid surface value within a
 *   small radius of it) + eye height, so it rises over buildings instead of
 *   passing through them. Over nodata (no valid value nearby) the last known
 *   ground is kept -- it never falls. Horizontally it is clamped to the
 *   terrain's bounding box; vertically to one diagonal above the highest point.
 * Raster values are only read, never changed.
 */
import { mapToPixel } from '../data/raster.js';
import { worldToMap } from './heightmap.js';

export const EYE_HEIGHT_FRACTION = 0.005;
export const SPEED_FRACTION = 0.04;
export const FAST_MULTIPLIER = 4;
export const LOOK_RAD_PER_PX = 0.004;
export const MAX_PITCH_DEG = 85;
const MAX_STEP_S = 0.1; // a long frame never becomes a teleport

const KEYS = { KeyW: 'forward', KeyS: 'back', KeyA: 'left', KeyD: 'right', KeyQ: 'down', KeyE: 'up', ShiftLeft: 'fast', ShiftRight: 'fast' };
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Eye height, speed and collision radius derived from the terrain's own extent. */
export function terrainScale(terrain) {
  const b = terrain.bounds();
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[2] - b.min[2]) || 1;
  const t = terrain.heightmap.transform;
  const pixel = Math.max(Math.hypot(t[0], t[3]), Math.hypot(t[1], t[4]));
  const eyeHeight = EYE_HEIGHT_FRACTION * diag;
  return { diag, eyeHeight, speed: SPEED_FRACTION * diag, collisionRadius: Math.max(0.5 * eyeHeight, pixel) };
}

/**
 * Highest valid surface (world y) within `radius` of world (x, z), from the
 * raster values; null when no valid sample is there.
 * @param {import('./heightmap.js').Heightmap} hm
 */
export function groundHeightAt(hm, x, z, radius) {
  const t = hm.transform;
  const [mx, my] = worldToMap(hm, x, z);
  const [c, r] = mapToPixel(t, mx, my);
  const rc = Math.ceil(radius / Math.hypot(t[0], t[3]));
  const rr = Math.ceil(radius / Math.hypot(t[1], t[4]));
  const col = Math.floor(c);
  const row = Math.floor(r);
  const { values, valid } = hm.raster;
  let best = -Infinity;
  for (let i = Math.max(0, row - rr); i <= Math.min(hm.height - 1, row + rr); i++) {
    for (let j = Math.max(0, col - rc); j <= Math.min(hm.width - 1, col + rc); j++) {
      const k = i * hm.width + j;
      if (valid[k]) best = Math.max(best, Number(values[k]));
    }
  }
  return Number.isFinite(best) ? best * hm.verticalScale : null;
}

/**
 * @param {{
 *   camera: import('three').PerspectiveCamera,
 *   domElement: HTMLElement,
 *   terrain: import('./terrain.js').TerrainLOD,
 *   requestFrame: () => void,
 *   schedule?: (fn: () => void) => void,
 *   now?: () => number,
 * }} options
 */
export function createFirstPerson({ camera, domElement, terrain, requestFrame, schedule = (fn) => requestAnimationFrame(fn), now = () => performance.now() }) {
  const hm = terrain.heightmap;
  const b = terrain.bounds();
  const scale = terrainScale(terrain);
  const ceiling = b.max[1] + scale.diag;
  const doc = domElement.ownerDocument;
  const held = new Set();
  let active = false;
  let yaw = 0;
  let pitch = 0;
  let lastGround = b.min[1];
  let ticking = false;
  let lastTick = 0;
  let look = null;

  const forward = () => [-Math.sin(yaw), 0, -Math.cos(yaw)];
  const right = () => [Math.cos(yaw), 0, -Math.sin(yaw)];

  function orient() {
    camera.rotation.set(pitch, yaw, 0, 'YXZ');
    camera.updateMatrixWorld();
  }

  /** Clamp into the terrain box and above the local surface. */
  function constrain() {
    const p = camera.position;
    p.x = clamp(p.x, b.min[0], b.max[0]);
    p.z = clamp(p.z, b.min[2], b.max[2]);
    const g = groundHeightAt(hm, p.x, p.z, scale.collisionRadius);
    if (g !== null) lastGround = g;
    const floor = lastGround + scale.eyeHeight;
    p.y = clamp(p.y, floor, Math.max(ceiling, floor));
  }

  /** Move by (forward, right, up) in world units along the current heading. */
  function move(f, r, u) {
    const F = forward();
    const R = right();
    camera.position.x += F[0] * f + R[0] * r;
    camera.position.z += F[2] * f + R[2] * r;
    camera.position.y += u;
    constrain();
  }

  /** Advance by dt seconds with the held keys. Returns true if the camera moved. */
  function step(dt) {
    if (!active) return false;
    const f = (held.has('forward') ? 1 : 0) - (held.has('back') ? 1 : 0);
    const r = (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
    const u = (held.has('up') ? 1 : 0) - (held.has('down') ? 1 : 0);
    if (!f && !r && !u) return false;
    const len = Math.hypot(f, r, u);
    const d = scale.speed * (held.has('fast') ? FAST_MULTIPLIER : 1) * Math.min(dt, MAX_STEP_S);
    move((f / len) * d, (r / len) * d, (u / len) * d);
    requestFrame();
    return true;
  }

  const moving = () => ['forward', 'back', 'left', 'right', 'up', 'down'].some((k) => held.has(k));
  function tick() {
    const t = now();
    step((t - lastTick) / 1000);
    lastTick = t;
    if (active && moving()) schedule(tick);
    else ticking = false;
  }

  const typing = (e) => /^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName ?? '');
  const onKeyDown = (e) => {
    const k = KEYS[e.code];
    if (!active || !k || typing(e)) return;
    e.preventDefault?.();
    held.add(k);
    if (!ticking && moving()) {
      ticking = true;
      lastTick = now();
      schedule(tick);
    }
  };
  const onKeyUp = (e) => {
    const k = KEYS[e.code];
    if (k) held.delete(k);
  };
  const onPointerDown = (e) => {
    if (active && e.button === 0) look = { x: e.clientX, y: e.clientY };
  };
  const onPointerMove = (e) => {
    if (!active || !look) return;
    lookBy(e.clientX - look.x, e.clientY - look.y);
    look = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = () => { look = null; };

  /** Turn by a mouse delta in pixels (drag right = look right, drag up = look up). */
  function lookBy(dx, dy) {
    yaw -= dx * LOOK_RAD_PER_PX;
    const lim = (MAX_PITCH_DEG * Math.PI) / 180;
    pitch = clamp(pitch - dy * LOOK_RAD_PER_PX, -lim, lim);
    orient();
    requestFrame();
  }

  doc.addEventListener('keydown', onKeyDown);
  doc.addEventListener('keyup', onKeyUp);
  domElement.addEventListener('pointerdown', onPointerDown);
  doc.addEventListener('pointermove', onPointerMove);
  doc.addEventListener('pointerup', onPointerUp);

  return {
    scale,
    get active() { return active; },
    /**
     * First-person pose standing at an orbit target, facing the way the orbit
     * camera was looking (horizontally), level.
     */
    poseFromOrbit(cameraPos, target) {
      const dx = target[0] - cameraPos[0];
      const dz = target[2] - cameraPos[2];
      return { position: [target[0], target[1], target[2]], yaw: dx || dz ? Math.atan2(-dx, -dz) : 0, pitch: 0 };
    },
    enable({ position, yaw: y0, pitch: p0 }) {
      active = true;
      yaw = y0;
      pitch = p0;
      camera.position.set(...position);
      lastGround = b.min[1];
      camera.position.y = -Infinity; // stand on the surface at this spot
      constrain();
      orient();
    },
    disable() {
      active = false;
      held.clear();
      look = null;
    },
    /** A point ahead of the camera on the terrain (for handing back to orbit). */
    pointAhead(distance) {
      const F = forward();
      const x = clamp(camera.position.x + F[0] * distance, b.min[0], b.max[0]);
      const z = clamp(camera.position.z + F[2] * distance, b.min[2], b.max[2]);
      const g = groundHeightAt(hm, x, z, scale.collisionRadius);
      return [x, g ?? lastGround, z];
    },
    move,
    step,
    lookBy,
    keyDown: (code) => onKeyDown({ code }),
    keyUp: (code) => onKeyUp({ code }),
    status() {
      return {
        yawDeg: (yaw * 180) / Math.PI,
        pitchDeg: (pitch * 180) / Math.PI,
        eyeHeight: scale.eyeHeight,
        speed: scale.speed,
        groundY: lastGround,
        aboveGround: camera.position.y - lastGround,
      };
    },
    dispose() {
      doc.removeEventListener('keydown', onKeyDown);
      doc.removeEventListener('keyup', onKeyUp);
      domElement.removeEventListener('pointerdown', onPointerDown);
      doc.removeEventListener('pointermove', onPointerMove);
      doc.removeEventListener('pointerup', onPointerUp);
    },
  };
}

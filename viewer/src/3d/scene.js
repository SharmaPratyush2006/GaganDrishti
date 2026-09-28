/**
 * Minimal Three.js scene: renderer, lights, and one fixed initial camera.
 *
 * In plain words: just enough to look at the terrain. There are NO user
 * camera controls in Step 3; the camera is placed deterministically from the
 * terrain's bounding box, south of the centre and looking down at ~35 degrees.
 * Lighting is a fixed hemisphere + directional light for shape only; it is
 * NOT the scene's physical sun.
 */
import * as THREE from 'three';

const CAMERA_ELEVATION_DEG = 35;
const CAMERA_DISTANCE_FACTOR = 0.9; // x the horizontal diagonal of the terrain

/**
 * Deterministic camera pose for a world AABB.
 * @param {{min:number[],max:number[]}} b
 * @returns {{position:number[], target:number[]}}
 */
export function initialCameraPose(b) {
  const target = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[2] - b.min[2]) || 1;
  const d = CAMERA_DISTANCE_FACTOR * diag;
  const el = (CAMERA_ELEVATION_DEG * Math.PI) / 180;
  // South of the centre (+z is south) looking north and down.
  return { position: [target[0], target[1] + d * Math.sin(el), target[2] + d * Math.cos(el)], target };
}

/**
 * @param {HTMLCanvasElement} canvas
 */
export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);
  // Neutral (grey/white) lights only, so a grayscale texture is not tinted into colour.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.9));
  const sunlike = new THREE.DirectionalLight(0xffffff, 1.6);
  sunlike.position.set(-1, 2, -0.6);
  scene.add(sunlike);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);

  let terrainGroup = null;

  function resize() {
    const w = canvas.clientWidth || 1;
    const h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  return {
    renderer,
    scene,
    camera,
    /** Place the fixed camera for a terrain AABB. */
    frame(bounds) {
      const { position, target } = initialCameraPose(bounds);
      const diag = Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]) || 1;
      camera.near = diag / 1000;
      camera.far = diag * 10;
      camera.position.set(...position);
      camera.lookAt(...target);
      camera.updateProjectionMatrix();
      return { position, target };
    },
    setTerrain(group) {
      if (terrainGroup) scene.remove(terrainGroup);
      terrainGroup = group;
      if (group) scene.add(group);
    },
    resize,
    render() {
      resize();
      renderer.render(scene, camera);
    },
    dispose() {
      if (terrainGroup) scene.remove(terrainGroup);
      renderer.dispose();
    },
  };
}

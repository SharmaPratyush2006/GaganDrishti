/**
 * TerrainLOD: the quadtree's selected tiles as Three.js meshes.
 *
 * In plain words: for a camera position, pick the tiles (quadtree.js), build
 * each tile's triangles (tileMesh.js) and show them. A tile's geometry is
 * cached by (tile, edge-stitching pattern), so changing the LOD re-uses tiles
 * that were built before instead of rebuilding the whole raster. When the
 * camera moves but the selection does not change, the shown meshes are left
 * exactly as they are.
 *
 * The texture (texture.js) only changes the material. Geometry -- and so every
 * height -- comes from the heightmap alone.
 *
 * Normals: computed from each tile's own triangles
 * (BufferGeometry.computeVertexNormals). Tiles do not share normals across
 * their borders, so faint shading seams can appear at tile edges; geometry
 * itself is crack-free (see tileMesh.js).
 */
import * as THREE from 'three';

import { buildQuadtree, nodeBounds, selectLod } from './quadtree.js';
import { buildTileMesh } from './tileMesh.js';

const LEVEL_TINTS = [0x5b7fa6, 0x6fa36b, 0xc9a24a, 0xc2645a, 0x9a6bc2, 0x4aa8a8, 0xb0b0b0];
export const BASE_COLOR = 0xb9c2cc;

/** @param {import('./tileMesh.js').TileMesh} tile */
export function tileGeometry(tile) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(tile.positions, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(tile.uvs, 2));
  // Raster index per vertex: lets a later step map any vertex back to its pixel.
  g.setAttribute('pixelIndex', new THREE.BufferAttribute(tile.pixelIndex, 1));
  g.setIndex(new THREE.BufferAttribute(tile.indices, 1));
  g.computeVertexNormals();
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

export class TerrainLOD {
  /**
   * @param {import('./heightmap.js').Heightmap} heightmap
   * @param {{tileSize?: number, lodFactor?: number}} [options]
   */
  constructor(heightmap, { tileSize = 64, lodFactor = 1.5 } = {}) {
    this.heightmap = heightmap;
    this.tileSize = tileSize;
    this.lodFactor = lodFactor;
    this.root = buildQuadtree(heightmap.width, heightmap.height, { tileSize });
    this.group = new THREE.Group();
    this.group.name = 'terrain';
    /** @type {Map<string, {geometry: THREE.BufferGeometry, tile: object}>} */
    this.cache = new Map();
    this.materials = LEVEL_TINTS.map((color) => new THREE.MeshLambertMaterial({ color }));
    this.baseMaterial = new THREE.MeshLambertMaterial({ color: BASE_COLOR });
    this.texturedMaterial = null;
    this.texture = null;
    this.tintByLevel = false;
    this.selectionKey = null;
    this.counters = { updates: 0, selectionChanges: 0, tilesBuiltTotal: 0 };
    this.lastStats = null;
  }

  /** World AABB of the whole terrain. */
  bounds() {
    return nodeBounds(this.heightmap, this.root);
  }

  materialFor(level) {
    if (this.tintByLevel) return this.materials[level % this.materials.length];
    return this.texturedMaterial ?? this.baseMaterial;
  }

  applyMaterials() {
    for (const mesh of this.group.children) mesh.material = this.materialFor(mesh.userData.level);
  }

  setTintByLevel(on) {
    this.tintByLevel = on;
    this.applyMaterials();
  }

  /**
   * Show a texture (THREE.Texture) on the surface, or none. Appearance only.
   * The caller owns the texture's lifetime.
   */
  setTexture(texture) {
    this.texturedMaterial?.dispose();
    this.texture = texture ?? null;
    this.texturedMaterial = texture ? new THREE.MeshLambertMaterial({ color: 0xffffff, map: texture }) : null;
    this.applyMaterials();
  }

  /**
   * Select tiles for a camera position and show them.
   * @param {number[]} camera  world [x, y, z]
   */
  update(camera) {
    const t0 = performance.now();
    this.counters.updates++;
    const { nodes, edges } = selectLod(this.heightmap, this.root, camera, { lodFactor: this.lodFactor });
    const keys = nodes.map((node) => {
      const e = edges.get(node);
      return { node, key: `${node.id}|${e.top},${e.bottom},${e.left},${e.right}`, e };
    });
    const selectionKey = keys.map((k) => k.key).join(';');
    const changed = selectionKey !== this.selectionKey;
    let built = 0;
    if (changed) {
      const meshes = keys.map(({ node, key, e }) => {
        let entry = this.cache.get(key);
        if (!entry) {
          const tile = buildTileMesh(this.heightmap, node, e);
          entry = { geometry: tileGeometry(tile), tile };
          this.cache.set(key, entry);
          built++;
        }
        const mesh = new THREE.Mesh(entry.geometry, this.materialFor(node.level));
        mesh.name = `tile ${node.id}`;
        mesh.userData = { nodeId: node.id, level: node.level, stride: node.stride, cacheKey: key };
        return mesh;
      });
      this.group.clear();
      if (meshes.length) this.group.add(...meshes);
      this.selectionKey = selectionKey;
      this.counters.selectionChanges++;
      this.counters.tilesBuiltTotal += built;
    }

    const byLevel = {};
    let vertices = 0;
    let triangles = 0;
    let dropped = 0;
    for (const m of this.group.children) {
      const { tile } = this.cache.get(m.userData.cacheKey);
      const k = `L${m.userData.level} (stride ${m.userData.stride})`;
      byLevel[k] = (byLevel[k] ?? 0) + 1;
      vertices += tile.vertexCount;
      triangles += tile.triangleCount;
      dropped += tile.droppedForNodata;
    }
    const { width, height } = this.heightmap;
    this.lastStats = {
      camera: [...camera],
      tileSize: this.tileSize,
      lodFactor: this.lodFactor,
      selectedTiles: this.group.children.length,
      tilesByLevel: byLevel,
      vertices,
      triangles,
      trianglesDroppedForNodata: dropped,
      fullResolutionTriangleUpperBound: 2 * Math.max(0, width - 1) * Math.max(0, height - 1),
      selectionChanged: changed,
      tilesBuiltThisUpdate: built,
      cachedGeometries: this.cache.size,
      ...this.counters,
      updateMs: performance.now() - t0,
    };
    return this.lastStats;
  }

  dispose() {
    for (const { geometry } of this.cache.values()) geometry.dispose();
    this.cache.clear();
    this.materials.forEach((m) => m.dispose());
    this.baseMaterial.dispose();
    this.texturedMaterial?.dispose();
    this.group.clear();
  }
}

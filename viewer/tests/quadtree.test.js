import { describe, expect, it } from 'vitest';

import { heightmapFromRaster, sampleXZ } from '../src/3d/heightmap.js';
import { allNodes, balance, buildQuadtree, selectLod, sharedEdge } from '../src/3d/quadtree.js';
import { buildTileMesh } from '../src/3d/tileMesh.js';
import { expected } from './helpers.js';
import { areaRC, grid, rasterFrom, trianglesRC } from './mesh3d.helpers.js';

const ABS = { transform: expected['tiny_ndsm_cog.tif'].transform, crs: 'EPSG:32643', units: 'metres' };

describe('quadtree structure', () => {
  for (const [w, h] of [[512, 512], [37, 21], [65, 65]]) {
    it(`root covers the full ${w}x${h} raster; children partition their parent`, () => {
      const root = buildQuadtree(w, h, { tileSize: 8 });
      expect([root.r0, root.c0, root.r1, root.c1]).toEqual([0, 0, h - 1, w - 1]);
      for (const n of allNodes(root)) {
        if (!n.children.length) {
          expect(n.stride).toBe(1);
          continue;
        }
        let area = 0;
        for (const ch of n.children) {
          expect(ch.r0).toBeGreaterThanOrEqual(n.r0);
          expect(ch.c0).toBeGreaterThanOrEqual(n.c0);
          expect(ch.r1).toBeLessThanOrEqual(n.r1);
          expect(ch.c1).toBeLessThanOrEqual(n.c1);
          expect(ch.level).toBe(n.level + 1);
          expect(ch.stride).toBe(Math.max(1, n.stride / 2));
          area += (ch.r1 - ch.r0) * (ch.c1 - ch.c0);
        }
        expect(area).toBe((n.r1 - n.r0) * (n.c1 - n.c0));
      }
    });
  }

  it('splits into four children where the raster allows', () => {
    const root = buildQuadtree(512, 512, { tileSize: 64 });
    expect(root.children.length).toBe(4);
    expect(root.stride).toBe(8);
    // 511 cells in a 512 span: every level still has 4 children down to stride 1.
    expect(allNodes(root).filter((n) => n.stride === 1).length).toBe(64);
  });
});

describe('LOD selection', () => {
  const raster = rasterFrom(grid(129, 129, (r, c) => Math.sin(r / 9) * 3 + Math.cos(c / 7) * 2), ABS);
  const hm = heightmapFromRaster(raster);
  const root = buildQuadtree(129, 129, { tileSize: 8 });
  const b = { far: [0, 5000, 5000], nearCorner: [...sampleXZ(hm, 0, 0)].flatMap((v, i) => (i === 0 ? [v, 5] : [v])) };

  it('a far camera draws the coarse root only', () => {
    const { nodes } = selectLod(hm, root, b.far);
    expect(nodes.map((n) => n.id)).toEqual(['r']);
    expect(nodes[0].stride).toBeGreaterThan(1);
  });

  it('a near camera selects full-resolution tiles near it and coarser ones away from it', () => {
    const { nodes } = selectLod(hm, root, b.nearCorner);
    const strides = new Set(nodes.map((n) => n.stride));
    expect(strides.has(1)).toBe(true);
    expect(strides.size).toBeGreaterThan(1);
    const nearest = nodes.find((n) => n.r0 === 0 && n.c0 === 0);
    expect(nearest.stride).toBe(1);
  });

  it('LOD changes the rendered resolution (triangle count), not just a stored tree', () => {
    const count = (cam) => {
      const { nodes, edges } = selectLod(hm, root, cam);
      return nodes.reduce((s, n) => s + buildTileMesh(hm, n, edges.get(n)).triangleCount, 0);
    };
    expect(count(b.nearCorner)).toBeGreaterThan(count(b.far));
  });

  it('is deterministic for a given camera', () => {
    const a = selectLod(hm, root, b.nearCorner).nodes.map((n) => n.id);
    const c = selectLod(hm, root, b.nearCorner).nodes.map((n) => n.id);
    expect(a).toEqual(c);
  });

  it('balances: edge-neighbours differ by at most one level', () => {
    const violations = (nodes) => {
      let n = 0;
      for (const x of nodes) for (const y of nodes) if (x !== y && sharedEdge(x, y) && Math.abs(x.level - y.level) > 1) n++;
      return n;
    };
    // Deliberately unbalanced: descend into the bottom-right quadrant, then keep
    // taking its top-left child, so deep tiles touch the coarse level-1 tiles.
    const path = [root, root.children[3]];
    while (path.at(-1).children.length) path.push(path.at(-1).children[0]);
    const unbalanced = [];
    for (let i = 0; i < path.length - 1; i++) unbalanced.push(...path[i].children.filter((c) => c !== path[i + 1]));
    unbalanced.push(path.at(-1));
    expect(violations(unbalanced)).toBeGreaterThan(0);
    const balanced = balance(unbalanced);
    expect(violations(balanced)).toBe(0);
    // Still an exact partition of the raster.
    expect(balanced.reduce((s, n) => s + (n.r1 - n.r0) * (n.c1 - n.c0), 0)).toBe(128 * 128);
    // And the camera-driven selection is balanced too.
    expect(violations(selectLod(hm, root, b.nearCorner, { lodFactor: 0.4 }).nodes)).toBe(0);
  });
});

describe('seams', () => {
  /**
   * Crack check in sample-index space over all selected tiles:
   *  - total triangle area equals the raster's cell area (no gap, no overlap), and
   *  - every edge is shared by exactly two triangles, except edges on the
   *    raster's outer border (one). A crack or T-junction breaks this.
   */
  function watertight(raster, root, cam, { stitch = true, lodFactor } = {}) {
    const hm = heightmapFromRaster(raster);
    const { nodes, edges } = selectLod(hm, root, cam, { lodFactor });
    const W = raster.width;
    const H = raster.height;
    let area = 0;
    const uses = new Map();
    let stitchedEdges = 0;
    for (const n of nodes) {
      const e = edges.get(n);
      stitchedEdges += ['top', 'bottom', 'left', 'right'].filter((k) => e[k] > n.stride).length;
      const tile = buildTileMesh(hm, n, stitch ? e : undefined);
      for (const t of trianglesRC(tile, W)) {
        area += Math.abs(areaRC(t));
        for (let k = 0; k < 3; k++) {
          const p = t[k];
          const q = t[(k + 1) % 3];
          const key = [p, q].map(([r, c]) => r * W + c).sort((x, y) => x - y).join('-');
          uses.set(key, (uses.get(key) ?? 0) + 1);
        }
      }
    }
    let bad = 0;
    for (const [key, n] of uses) {
      const [p, q] = key.split('-').map(Number).map((i) => [Math.floor(i / W), i % W]);
      const onBorder = (p[0] === q[0] && (p[0] === 0 || p[0] === H - 1)) || (p[1] === q[1] && (p[1] === 0 || p[1] === W - 1));
      if (n !== (onBorder ? 1 : 2)) bad++;
    }
    return { area, expectedArea: (W - 1) * (H - 1), badEdges: bad, stitchedEdges, strides: new Set(nodes.map((n) => n.stride)) };
  }

  const cases = [
    ['square 65x65', 65, 65],
    ['odd 50x37', 50, 37],
  ];
  for (const [name, w, h] of cases) {
    it(`mixed-LOD tiles are crack-free (${name})`, () => {
      const raster = rasterFrom(grid(h, w, (r, c) => (r * 31 + c * 17) % 11), ABS);
      const root = buildQuadtree(w, h, { tileSize: 4 });
      const hm = heightmapFromRaster(raster);
      for (const [r, c] of [[0, 0], [h - 1, w - 1], [Math.floor(h / 2), 3]]) {
        const [x, z] = sampleXZ(hm, r, c);
        const res = watertight(raster, root, [x, 2, z], { lodFactor: 0.6 });
        expect(res.strides.size).toBeGreaterThan(1); // LOD really mixes resolutions
        expect(res.stitchedEdges).toBeGreaterThan(0); // and coarse/fine edges really meet
        expect(res.badEdges).toBe(0);
        expect(res.area).toBe(res.expectedArea);
      }
    });
  }

  it('without stitching the same selection would crack (the test detects seams)', () => {
    const raster = rasterFrom(grid(65, 65, () => 1), ABS);
    const root = buildQuadtree(65, 65, { tileSize: 4 });
    const hm = heightmapFromRaster(raster);
    const [x, z] = sampleXZ(hm, 0, 0);
    const res = watertight(raster, root, [x, 2, z], { stitch: false, lodFactor: 0.6 });
    expect(res.badEdges).toBeGreaterThan(0);
  });
});

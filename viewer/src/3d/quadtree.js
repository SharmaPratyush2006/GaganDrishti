/**
 * Quadtree over the raster samples, and camera-distance LOD selection.
 *
 * In plain words: the raster is cut into square tiles, each tile into four
 * smaller tiles, and so on. Every tile is drawn with at most `tileSize` x
 * `tileSize` cells, so a big tile skips samples (its "stride") and a small tile
 * uses every sample. Tiles near the camera are split into their children
 * (finer); tiles far away are drawn whole (coarser).
 *
 * Geometry of a node (all in SAMPLE indices; sample = pixel centre):
 *   rows r0..r1, cols c0..c1 inclusive. Neighbouring nodes share their edge
 *   samples (a node's r1 is the next node's r0), so no gap exists between them.
 *   span   = nominal side length in cells, a power of two
 *   stride = max(1, span / tileSize): the node samples every stride-th pixel,
 *            aligned to multiples of stride over the whole raster
 *   The root has span = next power of two >= the raster's cell count and is
 *   clipped to the raster; children are clipped likewise.
 *
 * LOD criterion (deterministic for a given camera position):
 *   split a node when stride > 1 and
 *     distance(camera, node's world bounding box) < lodFactor * node's world diagonal
 *   Leaves (stride 1) are full resolution.
 *
 * Balance (restricted transitions): after selection, any selected node more
 * than one level coarser than an edge-neighbour is split, until none is. So an
 * edge only ever meets a neighbour of equal stride or twice its stride, which
 * the seam stitching in tileMesh.js relies on.
 */
import { sampleValid, sampleXZ } from './heightmap.js';

/**
 * @typedef {object} QuadNode
 * @property {string} id       path from the root, e.g. "r", "r2", "r21"
 * @property {number} level
 * @property {number} r0
 * @property {number} c0
 * @property {number} r1
 * @property {number} c1
 * @property {number} span
 * @property {number} stride
 * @property {QuadNode[]} children
 * @property {QuadNode|null} parent
 * @property {{min:number[],max:number[]}|null} [bounds]  world AABB, computed lazily
 */

const nextPow2 = (n) => 2 ** Math.ceil(Math.log2(Math.max(1, n)));

/**
 * @param {number} width   samples per row
 * @param {number} height  rows
 * @param {{tileSize?: number}} [options]  tileSize must be a power of two
 * @returns {QuadNode}
 */
export function buildQuadtree(width, height, { tileSize = 64 } = {}) {
  if (tileSize < 2 || nextPow2(tileSize) !== tileSize) throw new Error(`tileSize must be a power of two >= 2, got ${tileSize}`);
  const rootSpan = Math.max(nextPow2(Math.max(width - 1, height - 1)), 1);
  const make = (id, level, r0, c0, span, parent) => {
    const node = {
      id, level, r0, c0,
      r1: Math.min(r0 + span, height - 1),
      c1: Math.min(c0 + span, width - 1),
      span,
      stride: Math.max(1, span / tileSize),
      children: [],
      parent,
      bounds: null,
    };
    if (node.stride > 1) {
      const half = span / 2;
      let k = 0;
      for (const dr of [0, half]) {
        for (const dc of [0, half]) {
          // A child exists only if it holds at least one cell of the raster.
          if (r0 + dr < height - 1 && c0 + dc < width - 1) node.children.push(make(`${id}${k}`, level + 1, r0 + dr, c0 + dc, half, node));
          k++;
        }
      }
    }
    return node;
  };
  return make('r', 0, 0, 0, rootSpan, null);
}

/** Depth-first list of every node. */
export function allNodes(root) {
  const out = [];
  const walk = (n) => { out.push(n); n.children.forEach(walk); };
  walk(root);
  return out;
}

/**
 * World-space AABB of a node: corners through the transform, heights from
 * its valid samples (at full resolution, so coarse tiles are not under-bounded).
 * @param {import('./heightmap.js').Heightmap} hm
 * @param {QuadNode} node
 */
export function nodeBounds(hm, node) {
  if (node.bounds) return node.bounds;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const values = hm.raster.values;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (let r = node.r0; r <= node.r1; r++) {
    for (let c = node.c0; c <= node.c1; c++) {
      if (!sampleValid(hm, r, c)) continue;
      const v = Number(values[r * hm.width + c]) * hm.verticalScale;
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    }
  }
  for (const [r, c] of [[node.r0, node.c0], [node.r0, node.c1], [node.r1, node.c0], [node.r1, node.c1]]) {
    // Horizontal position only: the value at the corner may be nodata.
    const [x, z] = sampleXZ(hm, r, c);
    min[0] = Math.min(min[0], x); max[0] = Math.max(max[0], x);
    min[2] = Math.min(min[2], z); max[2] = Math.max(max[2], z);
  }
  // A node with no valid sample has no height extent; use 0-thickness at 0.
  min[1] = Number.isFinite(ymin) ? ymin : 0;
  max[1] = Number.isFinite(ymax) ? ymax : 0;
  node.bounds = { min, max, hasValid: Number.isFinite(ymin) };
  return node.bounds;
}

function distanceToBox(p, { min, max }) {
  let s = 0;
  for (let k = 0; k < 3; k++) {
    const d = p[k] < min[k] ? min[k] - p[k] : p[k] > max[k] ? p[k] - max[k] : 0;
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Horizontal world diagonal of a node. */
function worldDiagonal(b) {
  return Math.hypot(b.max[0] - b.min[0], b.max[2] - b.min[2]);
}

/**
 * Nodes to draw for a camera position (before balancing).
 * @param {import('./heightmap.js').Heightmap} hm
 * @param {QuadNode} root
 * @param {number[]} camera  world [x, y, z]
 * @param {{lodFactor?: number}} [options]
 * @returns {QuadNode[]}
 */
export function selectNodes(hm, root, camera, { lodFactor = 1.5 } = {}) {
  const out = [];
  const visit = (n) => {
    if (n.children.length && shouldSplit(hm, n, camera, lodFactor)) n.children.forEach(visit);
    else out.push(n);
  };
  visit(root);
  return out;
}

export function shouldSplit(hm, node, camera, lodFactor) {
  if (node.stride <= 1 || node.children.length === 0) return false;
  const b = nodeBounds(hm, node);
  return distanceToBox(camera, b) < lodFactor * worldDiagonal(b);
}

/**
 * Length of the shared boundary between two nodes, and on which side of `a`.
 * @returns {null | {side:'top'|'bottom'|'left'|'right'}}
 */
export function sharedEdge(a, b) {
  const rowOverlap = Math.min(a.r1, b.r1) - Math.max(a.r0, b.r0);
  const colOverlap = Math.min(a.c1, b.c1) - Math.max(a.c0, b.c0);
  if (colOverlap > 0 && a.r0 === b.r1) return { side: 'top' };
  if (colOverlap > 0 && a.r1 === b.r0) return { side: 'bottom' };
  if (rowOverlap > 0 && a.c0 === b.c1) return { side: 'left' };
  if (rowOverlap > 0 && a.c1 === b.c0) return { side: 'right' };
  return null;
}

/**
 * Split selected nodes until no edge-neighbours differ by more than one level.
 * @param {QuadNode[]} selected
 * @returns {QuadNode[]}  sorted by id (deterministic)
 */
export function balance(selected) {
  let nodes = [...selected];
  for (;;) {
    const toSplit = new Set();
    for (const a of nodes) {
      for (const b of nodes) {
        if (a !== b && b.level - a.level > 1 && sharedEdge(a, b)) toSplit.add(a);
      }
    }
    if (toSplit.size === 0) break;
    nodes = nodes.flatMap((n) => (toSplit.has(n) ? n.children : [n]));
  }
  return nodes.sort((x, y) => (x.id < y.id ? -1 : x.id > y.id ? 1 : 0));
}

/**
 * For each selected node, the stride of the coarser neighbour on each edge
 * (or the node's own stride when the neighbour is not coarser / absent).
 * @param {QuadNode[]} nodes  balanced selection
 * @returns {Map<QuadNode, {top:number,bottom:number,left:number,right:number}>}
 */
export function edgeStrides(nodes) {
  const out = new Map();
  for (const a of nodes) {
    const e = { top: a.stride, bottom: a.stride, left: a.stride, right: a.stride };
    for (const b of nodes) {
      if (a === b || b.stride <= a.stride) continue;
      const s = sharedEdge(a, b);
      if (s) e[s.side] = Math.max(e[s.side], b.stride);
    }
    out.set(a, e);
  }
  return out;
}

/**
 * Full LOD selection: select, balance, compute stitching.
 * @returns {{nodes: QuadNode[], edges: Map<QuadNode, object>}}
 */
export function selectLod(hm, root, camera, options) {
  const nodes = balance(selectNodes(hm, root, camera, options));
  return { nodes, edges: edgeStrides(nodes) };
}

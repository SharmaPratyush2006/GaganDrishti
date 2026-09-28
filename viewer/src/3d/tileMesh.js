/**
 * One quadtree tile -> indexed triangle mesh (plain typed arrays, no WebGL).
 *
 * In plain words: a tile takes every stride-th sample in its area, and every
 * little square between four samples becomes two triangles. Vertices are
 * shared (indexed), not duplicated per triangle.
 *
 * Triangulation of a cell with corners A=(r,c) B=(r,c') C=(r',c) D=(r',c'):
 *   (A, C, B) and (B, C, D)    -- upward-facing for a north-up transform;
 *   winding is reversed when the transform mirrors the grid.
 *
 * Nodata -- a triangle is emitted only if nothing under it is nodata:
 *   full-resolution cell (stride 1, no stitching): its 3 vertices are valid;
 *   any other triangle: every raster sample inside its bounding box is valid.
 * So nodata is never bridged; coarse tiles show *larger* holes near nodata,
 * never invented surface. Invalid samples never become vertices.
 *
 * Seams -- vertex snapping onto the coarser neighbour's samples:
 *   On an edge whose neighbour has a larger stride (always exactly 2x after
 *   balancing), each edge vertex that the neighbour does not sample is moved
 *   to the previous sample the neighbour does have. The tile's edge then
 *   consists of exactly the neighbour's edge segments: no T-junction, no
 *   crack. Triangles that collapse (two identical corners) are dropped; the
 *   remaining ones form a fan that covers the cell pair exactly.
 */
import { blockValid, samplePosition, sampleValid } from './heightmap.js';

/** Multiples of `stride` in [start, end), then `end`. `start` is a multiple of stride. */
export function sampleIndices(start, end, stride) {
  const out = [];
  for (let v = start; v < end; v += stride) out.push(v);
  out.push(end);
  return out;
}

/** Move `v` onto the coarse sampling (multiples of ns, plus `end`). */
function snapAlong(v, ns, end) {
  return v === end || v % ns === 0 ? v : v - (v % ns);
}

/**
 * @typedef {object} TileMesh
 * @property {Float32Array} positions   xyz per vertex
 * @property {Float32Array} uvs         texel centre of the vertex's pixel: ((col+0.5)/W, (row+0.5)/H)
 * @property {Uint32Array} indices      3 per triangle
 * @property {Uint32Array} pixelIndex   raster index (row * width + col) per vertex
 * @property {number} vertexCount
 * @property {number} triangleCount
 * @property {number} droppedForNodata  triangles omitted because they touch nodata
 */

/**
 * @param {import('./heightmap.js').Heightmap} hm
 * @param {import('./quadtree.js').QuadNode} node
 * @param {{top:number,bottom:number,left:number,right:number}} [edges]  neighbour strides
 * @returns {TileMesh}
 */
export function buildTileMesh(hm, node, edges) {
  const { r0, c0, r1, c1, stride } = node;
  const e = edges ?? { top: stride, bottom: stride, left: stride, right: stride };
  const rows = sampleIndices(r0, r1, stride);
  const cols = sampleIndices(c0, c1, stride);
  const W = hm.width;

  const snap = (r, c) => {
    let rr = r;
    let cc = c;
    if (rr === r0 && e.top > stride) cc = snapAlong(cc, e.top, c1);
    else if (rr === r1 && e.bottom > stride) cc = snapAlong(cc, e.bottom, c1);
    if (cc === c0 && e.left > stride) rr = snapAlong(rr, e.left, r1);
    else if (cc === c1 && e.right > stride) rr = snapAlong(rr, e.right, r1);
    return [rr, cc];
  };

  const vertexOf = new Map(); // raster index -> local vertex index
  const pos = [];
  const uv = [];
  const pix = [];
  const idx = [];
  let dropped = 0;
  const vertex = (r, c) => {
    const key = r * W + c;
    let v = vertexOf.get(key);
    if (v === undefined) {
      v = pix.length;
      vertexOf.set(key, v);
      pix.push(key);
      pos.push(...samplePosition(hm, r, c));
      // Texel centre of the same-grid source image (see texture.js for orientation).
      uv.push((c + 0.5) / W, (r + 0.5) / hm.height);
    }
    return v;
  };
  const triangle = (p, q, s, exact) => {
    const keys = [p, q, s].map(([r, c]) => r * W + c);
    if (keys[0] === keys[1] || keys[1] === keys[2] || keys[0] === keys[2]) return; // collapsed by stitching
    const ok = exact
      ? sampleValid(hm, p[0], p[1]) && sampleValid(hm, q[0], q[1]) && sampleValid(hm, s[0], s[1])
      : blockValid(hm, Math.min(p[0], q[0], s[0]), Math.min(p[1], q[1], s[1]), Math.max(p[0], q[0], s[0]), Math.max(p[1], q[1], s[1]));
    if (!ok) {
      dropped++;
      return;
    }
    const a = vertex(...p);
    const b = vertex(...q);
    const c = vertex(...s);
    if (hm.flipWinding) idx.push(a, c, b);
    else idx.push(a, b, c);
  };

  for (let i = 0; i < rows.length - 1; i++) {
    for (let j = 0; j < cols.length - 1; j++) {
      const A = [rows[i], cols[j]];
      const B = [rows[i], cols[j + 1]];
      const C = [rows[i + 1], cols[j]];
      const D = [rows[i + 1], cols[j + 1]];
      const [sA, sB, sC, sD] = [snap(...A), snap(...B), snap(...C), snap(...D)];
      const moved = sA[0] !== A[0] || sA[1] !== A[1] || sB[0] !== B[0] || sB[1] !== B[1]
        || sC[0] !== C[0] || sC[1] !== C[1] || sD[0] !== D[0] || sD[1] !== D[1];
      const exact = stride === 1 && !moved;
      triangle(sA, sC, sB, exact);
      triangle(sB, sC, sD, exact);
    }
  }

  return {
    positions: Float32Array.from(pos),
    uvs: Float32Array.from(uv),
    indices: Uint32Array.from(idx),
    pixelIndex: Uint32Array.from(pix),
    vertexCount: pix.length,
    triangleCount: idx.length / 3,
    droppedForNodata: dropped,
  };
}

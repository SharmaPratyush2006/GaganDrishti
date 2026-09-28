import { formatValue, PRODUCT_MEANINGS } from '../data/metadata.js';

const fmtMs = (ms) => (ms == null ? 'n/a' : `${ms.toFixed(1)} ms`);
const vec = (v) => `[${v.map((x) => x.toFixed(1)).join(', ')}]`;
const fp = (camera) => camera.firstPerson;

/** Debug panel: what is meshed, how it is textured, what the LOD selected, where the camera is. */
export default function MeshDebugPanel({ info }) {
  if (!info) return null;
  const { heightmap: hm, lod, camera, texture, timings } = info;
  const r = hm.raster;
  const t = timings;
  const rows = [
    ['product', `${r.productType} — ${PRODUCT_MEANINGS[r.productType]}`],
    ['source', r.source],
    ['mode', `${r.mode} (${r.modeReason})`],
    ['raster size', `${r.width} × ${r.height} samples`],
    ['valid / nodata samples', `${hm.stats.validPixels.toLocaleString()} / ${hm.stats.nodataPixels.toLocaleString()}`],
    ['value range', `${formatValue(hm.stats.min, r, 3)} … ${formatValue(hm.stats.max, r, 3)}`],
    ['horizontal axes', hm.horizontalUnits],
    ['vertical scale', `×${hm.verticalScale.toPrecision(4)} — ${hm.verticalScaleReason}`],
    ['texture', texture.label],
    ['source image', texture.source.kind === 'missing' ? 'none linked' : `${texture.source.label}${texture.source.kind === 'grayscale' ? ' — not RGB' : ''}`],
    ['image / raster grids', texture.alignment ? (texture.alignment.aligned ? 'aligned (same size, transform and CRS)' : `NOT aligned: ${texture.alignment.reasons.join('; ')}`) : 'n/a'],
    ['image nodata texels', texture.status === 'unavailable' ? 'n/a' : `${texture.nodataTexels.toLocaleString()} (drawn in the untextured surface colour)`],
    ['LOD criterion', `split while distance(camera, tile box) < ${lod.lodFactor} × tile diagonal; ${lod.tileSize}×${lod.tileSize} cells per tile; neighbours ≤ 1 level apart`],
    ['selected tiles', `${lod.selectedTiles}: ${Object.entries(lod.tilesByLevel).map(([k, v]) => `${v} × ${k}`).join(', ')}`],
    ['mesh vertices', lod.vertices.toLocaleString()],
    ['mesh triangles', `${lod.triangles.toLocaleString()} (full resolution would be ≤ ${lod.fullResolutionTriangleUpperBound.toLocaleString()})`],
    ['triangles omitted for nodata', lod.trianglesDroppedForNodata.toLocaleString()],
    ['LOD updates', `${lod.updates} updates, ${lod.selectionChanges} selection changes, ${lod.tilesBuiltTotal} tiles built in total, ${lod.cachedGeometries} cached; last update built ${lod.tilesBuiltThisUpdate} in ${fmtMs(lod.updateMs)}`],
    ['camera mode', camera.mode === 'firstPerson' ? 'First Person' : 'Orbit'],
    ['controls', camera.mode === 'firstPerson'
      ? `first person — eye height ${fp(camera).eyeHeight.toFixed(2)}, speed ${fp(camera).speed.toFixed(1)}/s (×4 with Shift), kept inside the terrain box and above the surface`
      : `OrbitControls — distance ${camera.limits.minDistance.toFixed(2)} … ${camera.limits.maxDistance.toFixed(0)}, polar ≤ ${camera.limits.maxPolarDeg}°, target kept over the terrain`],
    ['camera', camera.mode === 'firstPerson'
      ? `position ${vec(camera.position)}; heading ${fp(camera).yawDeg.toFixed(1)}°; pitch ${fp(camera).pitchDeg.toFixed(1)}°; ${fp(camera).aboveGround.toFixed(2)} above the surface below`
      : `position ${vec(camera.position)} → target ${vec(camera.target)}; distance ${camera.distance.toFixed(1)}; polar ${camera.polarDeg.toFixed(1)}°; azimuth ${camera.azimuthDeg.toFixed(1)}°`],
    ['control events', `${camera.changeEvents} orbit change events, ${camera.modeSwitches} mode switches → ${camera.frames} LOD/render frames`],
    ['measured this run', `heightmap ${fmtMs(t.heightmapMs)}; quadtree+texture ${fmtMs(t.quadtreeAndTextureMs)} (texture build ${fmtMs(texture.buildMs)}); first LOD+render ${fmtMs(t.firstLodAndRenderMs)}`],
  ];
  return (
    <dl className="mesh-debug" data-testid="mesh-debug">
      {rows.map(([k, v]) => (
        <div key={k} data-stat={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
